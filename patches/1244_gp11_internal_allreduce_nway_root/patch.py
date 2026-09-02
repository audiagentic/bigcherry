"""GP11 (preliminary): extend the internal pinned-host AllReduce pipeline to
support a 3-device fused reduce-to-root + broadcast path.

Patch 1001_hip_internal_allreduce's pipeline (allreduce.cu) hard-requires
``n_devices == 2`` -- a real pairwise ping-pong scheme, not a general N-way
algorithm. This project's own base-level test harness
(tools/lab/gp10-collective-harness/nway_star_allreduce.cpp) found, on real
hardware (2x RX 7900 XTX + 1x Radeon AI PRO R9700), that a naive "star"
N-way extension (every rank reads every other rank) scales roughly linearly
in N-1 and does not beat RCCL at N=3 -- but a fused reduce-to-root +
broadcast design (leaves each do exactly one handshake with a single root
rank; root reduces and republishes for leaves to read back), after fixing a
real cross-device pointer-alias bug the harness had and applying an
independent-float4-load optimization to root's reduce step, reliably BEATS
RCCL's own Ring algorithm by ~22% at decode-realistic sizes (30720 F32
elements: 68.3-68.8us vs RCCL's 87.8us, either RX 7900 XTX as root, 800+
stress reps with zero correctness failures). At large/prefill sizes
(2621440 elements) it still loses to RCCL, matching patch 1001's own
existing 2-device design where large messages are meant to stay on RCCL.

This patch wires that validated mechanism into ggml-cuda's real dispatch
path as a genuinely new N=3 code path, alongside (not replacing) the
existing N=2 pairwise mechanism:

* ``ggml_cuda_ar_pipeline_init`` now accepts n_devices == 2 (unchanged) or
  == 3 (new). For n_devices == 3, BF16 wire compression is forced off
  (bf16_threshold = 0) -- the validated regime is raw F32 wire transfer
  only, no compression has been tested for N=3 yet.
* A new device kernel ``ggml_cuda_ar_kernel3`` implements the fused
  root+broadcast protocol (F32 only, float4-vectorized, 8-block grid
  matching the existing GGML_CUDA_AR_KERNEL_BLOCKS), and a new host-side
  helper ``ggml_cuda_ar_allreduce_root3`` dispatches it per chunk, reusing
  the existing arrival-slot/event-pool machinery unchanged.
* ``ggml_cuda_ar_allreduce`` gains an early, self-contained n==3 branch
  (F32-only, chunked-kernel-eligible sizes only -- falls back to the
  caller's next provider, i.e. RCCL, for non-F32 types or large/prefill
  sizes) that returns before reaching any of the existing N=2-only code
  (BF16 conversion, copy-engine path, the pairwise chunked-kernel loop).
  The existing N=2 path is completely untouched -- zero regression risk.
* Root is always ``devices[0]`` -- both real RX 7900 XTX devices validated
  as root on this box; a future item (not this patch) could make root
  choice topology-aware if a different real topology needs it.

PRELIMINARY: this is deliberately narrower than the validated N=2 path
(no BF16 wire compression, no F16/BF16 native-type support, large messages
not implemented). Real production wiring (admission into 0840's hybrid
dispatch alongside GP01's qualified {0,1,2} topology, and a real GP08
full-stack MTP completion-bench validation -- this harness only measures
raw collective time, not the real end-to-end dispatch-path overhead RCCL
pays) is tracked separately in GP11's plan notes, not this patch.
"""

GROUP = "gpu-collectives"
STATE = "untested"

from bigcherry.patcher import Edit, FilePatch

_KERNEL3_TEXT = '''
// ---------------------------------------------------------------------------
// N=3 fused reduce-to-root + broadcast (GP11, preliminary). devices[0] is
// always root -- real hardware validated (either RX 7900 XTX as root beats
// RCCL's own Ring by ~22% at decode-realistic sizes on this box; see
// tools/lab/gp10-collective-harness/ and docs/planning/.../gpu-collectives/
// GP11.md). F32 only, no BF16 wire compression (bf16_threshold is forced
// to 0 for n_devices==3 in pipeline_init) -- narrower than the validated
// N=2 path, intentionally: this is exactly the regime GP11's own
// base-level harness tested before this patch was written.
//
// One kernel body, launched once per device with per-device pointers --
// the same pattern the N=2 kernel above uses. Root reads both leaves'
// pinned slots, sums, republishes into its own slot; leaves each do
// exactly one handshake with root only (not with each other).
// ---------------------------------------------------------------------------
static __global__ void ggml_cuda_ar_kernel3(
        const float * sendbuf,
        float       * recvbuf,
        float       * __restrict__ host_mine,
        const float * __restrict__ host_leaf0,   // valid only when is_root
        const float * __restrict__ host_leaf1,   // valid only when is_root
        const float * __restrict__ host_root,    // valid only when !is_root
        int                         count,
        int *                       arrival_mine, // BASE (slot,rank) pointer -- this kernel
        int *                       arrival_a,    // adds the per-block offset itself, mirroring
        int *                       arrival_b,    // ggml_cuda_ar_kernel's own per-block scheme
        int                         token,
        bool                        is_root) {
    const int gtid    = blockIdx.x * blockDim.x + threadIdx.x;
    const int gnt      = gridDim.x * blockDim.x;
    constexpr int ELEMS_PER_VEC = 4;
    const int count4  = count / ELEMS_PER_VEC;
    const int tail     = count4 * ELEMS_PER_VEC;

    // 2026-09-02 fix (gpt-dev-agent root cause, req_1f050a8abef749a2): the
    // original version signalled/polled the BASE (slot,rank) pointer
    // directly, so all GGML_CUDA_AR_KERNEL_BLOCKS blocks shared one token --
    // any block finishing first released every other block's wait, which
    // then read stripes other blocks hadn't published yet. Real hardware
    // symptom: ran fast, no crash, garbage output. ggml_cuda_ar_kernel above
    // (the working N=2 path) already gets this right via
    // ggml_cuda_ar_arrival_ptr's block-indexed slots; this kernel must do
    // the equivalent block-offset itself since it's handed BASE pointers.
    constexpr int ARRIVAL_INTS = (int) (GGML_CUDA_AR_ARRIVAL_STRIDE / sizeof(int));
    int * my_slot = arrival_mine + blockIdx.x * ARRIVAL_INTS;
    int * a_slot  = arrival_a ? arrival_a + blockIdx.x * ARRIVAL_INTS : nullptr;
    int * b_slot  = arrival_b ? arrival_b + blockIdx.x * ARRIVAL_INTS : nullptr;

    if (!is_root) {
        const float4 * src4 = reinterpret_cast<const float4 *>(sendbuf);
        float4       * dst4 = reinterpret_cast<float4 *>(host_mine);
        for (int i = gtid; i < count4; i += gnt) {
            dst4[i] = src4[i];
        }
        if (blockIdx.x == 0 && threadIdx.x < count - tail) {
            host_mine[tail + threadIdx.x] = sendbuf[tail + threadIdx.x];
        }
        __threadfence_system();
        __syncthreads();
        if (threadIdx.x == 0) {
            ggml_cuda_ar_signal_set(my_slot, token);
            __threadfence_system();
        }
    } else {
        __syncthreads();
    }

    if (threadIdx.x == 0 && is_root) {
        while (ggml_cuda_ar_signal_get(a_slot) != token) {
#ifdef GGML_USE_HIP
            __builtin_amdgcn_s_sleep(4);
#elif __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
            __nanosleep(100);
#else
            NO_DEVICE_CODE;
#endif
        }
        while (ggml_cuda_ar_signal_get(b_slot) != token) {
#ifdef GGML_USE_HIP
            __builtin_amdgcn_s_sleep(4);
#elif __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
            __nanosleep(100);
#else
            NO_DEVICE_CODE;
#endif
        }
    }
    __syncthreads();
    __threadfence_system();

    if (is_root) {
        const float4 * src4 = reinterpret_cast<const float4 *>(sendbuf);
        const float4 * l04  = reinterpret_cast<const float4 *>(host_leaf0);
        const float4 * l14  = reinterpret_cast<const float4 *>(host_leaf1);
        float4       * dst4 = reinterpret_cast<float4 *>(recvbuf);
        float4       * pub4 = reinterpret_cast<float4 *>(host_mine);
        for (int i = gtid; i < count4; i += gnt) {
            const float4 a = l04[i];
            const float4 b = l14[i];
            const float4 s = src4[i];
            float4 out;
            out.x = s.x + a.x + b.x;
            out.y = s.y + a.y + b.y;
            out.z = s.z + a.z + b.z;
            out.w = s.w + a.w + b.w;
            dst4[i] = out;
            pub4[i] = out;
        }
        if (blockIdx.x == 0 && threadIdx.x < count - tail) {
            const int i = tail + threadIdx.x;
            const float out = sendbuf[i] + host_leaf0[i] + host_leaf1[i];
            recvbuf[i]   = out;
            host_mine[i] = out;
        }
        __threadfence_system();
        __syncthreads();
        if (threadIdx.x == 0) {
            ggml_cuda_ar_signal_set(my_slot, token);
            __threadfence_system();
        }
    } else {
        if (threadIdx.x == 0) {
            while (ggml_cuda_ar_signal_get(a_slot) != token) {
#ifdef GGML_USE_HIP
                __builtin_amdgcn_s_sleep(4);
#elif __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
                __nanosleep(100);
#else
                NO_DEVICE_CODE;
#endif
            }
        }
        __syncthreads();
        __threadfence_system();
        const float4 * root4 = reinterpret_cast<const float4 *>(host_root);
        float4       * dst4  = reinterpret_cast<float4 *>(recvbuf);
        for (int i = gtid; i < count4; i += gnt) {
            dst4[i] = root4[i];
        }
        if (blockIdx.x == 0 && threadIdx.x < count - tail) {
            recvbuf[tail + threadIdx.x] = host_root[tail + threadIdx.x];
        }
    }
}
'''

_ROOT3_HELPER_TEXT = '''
// N=3 arrival-slot base pointer, resolved from CONSUMER c's own cross-device
// alias table (p->arrival_dev3[c]) rather than the single shared p->arrival.dev
// the N=2 path uses -- see add-n3-alias-fields/resolve-n3-aliases-at-init.
// Returns the BASE (slot,rank) pointer; ggml_cuda_ar_kernel3 adds the
// per-block offset itself (same split as ggml_cuda_ar_arrival_ptr/
// ggml_cuda_ar_kernel above).
static int * ggml_cuda_ar_arrival_ptr3(
        const ggml_cuda_ar_pipeline * p, int consumer, int slot, int rank) {
    const size_t offset = ((size_t) slot * 3 + rank) *
                          GGML_CUDA_AR_KERNEL_BLOCKS * GGML_CUDA_AR_ARRIVAL_STRIDE;
    return reinterpret_cast<int *>(
        reinterpret_cast<uint8_t *>(p->arrival_dev3[consumer]) + offset);
}

// N=3 dispatch helper (GP11, preliminary) -- see ggml_cuda_ar_kernel3 above.
// devices[0] is always root. Reuses the existing per-slot chunking/
// event-pool machinery (ggml_cuda_ar_acquire_slot) unchanged; pointer
// aliasing goes through the N=3-specific per-consumer tables above instead
// of ggml_cuda_ar_arrival_ptr/host_buf[owner].dev (those are only valid
// from the OWNING device, not every consumer -- see add-n3-alias-fields).
static bool ggml_cuda_ar_allreduce_root3(
        ggml_cuda_ar_pipeline * p,
        ggml_backend_t        * backends,
        ggml_tensor           ** tensors,
        int64_t                 ne) {
    GGML_ASSERT(p->n_devices == 3);
    const size_t max_chunk_elems = p->buf_bytes / sizeof(float);
    GGML_ASSERT(max_chunk_elems > 0);

    bool compute_flag[3];
    for (int i = 0; i < 3; ++i) {
        compute_flag[i] = (tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) != 0;
    }

    for (int64_t chunk_start = 0; chunk_start < ne; chunk_start += (int64_t) max_chunk_elems) {
        const size_t remaining_elems = (size_t) (ne - chunk_start);
        const size_t chunk_elems = remaining_elems < max_chunk_elems ? remaining_elems : max_chunk_elems;
        const size_t chunk_bytes = chunk_elems * sizeof(float);

        const auto [slot, token] = ggml_cuda_ar_acquire_slot(p);

        for (int i = 0; i < 3; ++i) {
            ggml_cuda_set_device(p->devices[i]);
            auto * cuda_ctx = static_cast<ggml_backend_cuda_context *>(backends[i]->context);
            GGML_ASSERT(cuda_ctx->device == p->devices[i]);
            cudaStream_t stream = cuda_ctx->stream();

            float * data = reinterpret_cast<float *>(tensors[i]->data) + chunk_start;

            if (!compute_flag[i]) {
                CUDA_CHECK(cudaMemsetAsync(data, 0, chunk_bytes, stream));
            }

            const bool is_root = (i == 0);
            float * host_mine_dev = reinterpret_cast<float *>(
                p->host_buf_dev3[i][i] + (size_t) slot * p->buf_bytes);

            if (is_root) {
                float * host0_dev = reinterpret_cast<float *>(
                    p->host_buf_dev3[i][1] + (size_t) slot * p->buf_bytes);
                float * host1_dev = reinterpret_cast<float *>(
                    p->host_buf_dev3[i][2] + (size_t) slot * p->buf_bytes);
                ggml_cuda_ar_kernel3<<<dim3(GGML_CUDA_AR_KERNEL_BLOCKS), dim3(256), 0, stream>>>(
                    data, data, host_mine_dev, host0_dev, host1_dev, nullptr,
                    (int) chunk_elems,
                    ggml_cuda_ar_arrival_ptr3(p, i, slot, 0),
                    ggml_cuda_ar_arrival_ptr3(p, i, slot, 1),
                    ggml_cuda_ar_arrival_ptr3(p, i, slot, 2),
                    token, true);
            } else {
                float * hostroot_dev = reinterpret_cast<float *>(
                    p->host_buf_dev3[i][0] + (size_t) slot * p->buf_bytes);
                ggml_cuda_ar_kernel3<<<dim3(GGML_CUDA_AR_KERNEL_BLOCKS), dim3(256), 0, stream>>>(
                    data, data, host_mine_dev, nullptr, nullptr, hostroot_dev,
                    (int) chunk_elems,
                    ggml_cuda_ar_arrival_ptr3(p, i, slot, i),
                    ggml_cuda_ar_arrival_ptr3(p, i, slot, 0),
                    nullptr,
                    token, false);
            }
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventRecord(p->ev_pool[i][slot].ker, stream));
        }
    }
    return true;
}
'''

CU = FilePatch(
    path="ggml/src/ggml-cuda/allreduce.cu",
    description="GP11 preliminary: N=3 fused reduce-to-root AllReduce path",
    edits=(
        Edit(
            id="pipeline-init-accept-n3",
            anchor=(
                r"    if \(n_devices != 2\) \{\n"
                r"        GGML_LOG_DEBUG\([^\n]*\n"
                r"[^\n]*\n"
                r"        return nullptr;\n"
                r"    \}"
            ),
            rationale="relax the pipeline_init guard to also accept 3 devices "
                      "(GP11's validated fused reduce-to-root path)",
            mode="replace",
            text=(
                "    if (n_devices != 2 && n_devices != 3) {\n"
                "        GGML_LOG_DEBUG(\"%s: internal AllReduce only supports n_devices=2 or 3 "
                "(got %zu); \"\n"
                "                       \"falling back\\n\", __func__, n_devices);\n"
                "        return nullptr;\n"
                "    }"
            ),
            guard=r"if \(n_devices != 2 && n_devices != 3\)",
        ),
        Edit(
            id="force-disable-bf16-for-n3",
            anchor=r"    p->bf16_threshold   = ggml_cuda_ar_env_u64\([^\n]*\);",
            rationale="N=3's validated regime is raw F32 wire transfer only -- "
                      "force BF16 wire compression off regardless of env override "
                      "until it's validated for N=3",
            mode="insert_after",
            text=(
                "\n"
                "    if (n_devices == 3) {\n"
                "        // GP11 preliminary: validated raw F32 wire transfer only.\n"
                "        p->bf16_threshold = 0;\n"
                "    }"
            ),
            guard=r"if \(n_devices == 3\) \{\n        // GP11 preliminary",
        ),
        Edit(
            id="insert-kernel3",
            anchor=(
                r"template <typename T_dst, typename T_src>\n"
                r"static __global__ void ggml_cuda_ar_add_kernel\("
            ),
            rationale="insert the new N=3 fused root+broadcast device kernel "
                      "right before the existing combined load-convert-add kernel",
            mode="insert_before",
            text=_KERNEL3_TEXT + "\n",
            guard=r"static __global__ void ggml_cuda_ar_kernel3\(",
        ),
        Edit(
            id="insert-root3-helper",
            anchor=(
                r"template <typename T_src, typename T_dst>\n"
                r"static bool ggml_cuda_ar_allreduce_copy_impl\("
            ),
            rationale="insert the N=3 host-side dispatch helper right before the "
                      "existing copy-engine helper, after ggml_cuda_ar_acquire_slot "
                      "and ggml_cuda_ar_arrival_ptr are already defined",
            mode="insert_before",
            text=_ROOT3_HELPER_TEXT + "\n",
            guard=r"static bool ggml_cuda_ar_allreduce_root3\(",
        ),
        Edit(
            id="add-n3-alias-fields",
            anchor=(
                r"    ggml_cuda_ar_host_mapping arrival;\n"
                r"\};"
            ),
            rationale="2026-09-02 fix (gpt-dev-agent, req_1f050a8abef749a2): the "
                      "original N=3 helper reused host_buf[owner].dev / arrival.dev "
                      "aliases resolved under whichever device happened to be "
                      "current in pipeline_init -- the exact same cross-device "
                      "pointer-alias bug already fixed in the harness "
                      "(tools/lab/gp10-collective-harness/). hipHostGetDevicePointer "
                      "must be called with the CONSUMING device current; store a "
                      "separate alias table per consumer, resolved once at init",
            mode="replace",
            text=(
                "    ggml_cuda_ar_host_mapping arrival;\n"
                "\n"
                "    // N=3 preliminary (GP11): per-consuming-device resolved aliases.\n"
                "    // [consumer][owner] -- host_buf_dev3[c][o] is the device pointer\n"
                "    // valid when dereferenced from consumer c's own kernel, for\n"
                "    // owner o's pinned stage buffer. Resolved once at init (not per\n"
                "    // call) since hipHostGetDevicePointer must be called with the\n"
                "    // consuming device current to be valid cross-device.\n"
                "    uint8_t * host_buf_dev3[3][3] = {};\n"
                "    int     * arrival_dev3[3]     = {};\n"
                "};"
            ),
            guard=r"uint8_t \* host_buf_dev3\[3\]\[3\] = \{\};",
        ),
        Edit(
            id="resolve-n3-aliases-at-init",
            anchor=(
                r"        if \(p->host_buf\[i\]\.alloc\(host_buf_total\) != cudaSuccess\) \{\n"
                r"[^\n]*\n"
                r"[^\n]*\n"
                r"            ggml_cuda_ar_pipeline_free\(p\);\n"
                r"            return nullptr;\n"
                r"        \}\n"
                r"    \}"
            ),
            rationale="resolve every N=3 cross-device pointer alias once, right "
                      "after host_buf[] is allocated, with each consuming device "
                      "current in turn",
            mode="insert_after",
            text=(
                "\n"
                "\n"
                "    if (n_devices == 3) {\n"
                "        for (int c = 0; c < 3; ++c) {\n"
                "            ggml_cuda_set_device(p->devices[c]);\n"
                "            void * arrival_alias = nullptr;\n"
                "            if (cudaHostGetDevicePointer(&arrival_alias, p->arrival.host, 0) != cudaSuccess) {\n"
                "                GGML_LOG_ERROR(\"%s: N=3 arrival alias resolution failed "
                "for consumer %d\\n\", __func__, c);\n"
                "                ggml_cuda_ar_pipeline_free(p);\n"
                "                return nullptr;\n"
                "            }\n"
                "            p->arrival_dev3[c] = reinterpret_cast<int *>(arrival_alias);\n"
                "            for (int owner = 0; owner < 3; ++owner) {\n"
                "                void * stage_alias = nullptr;\n"
                "                if (cudaHostGetDevicePointer(&stage_alias, p->host_buf[owner].host, 0) != cudaSuccess) {\n"
                "                    GGML_LOG_ERROR(\"%s: N=3 stage alias resolution failed "
                "for consumer %d owner %d\\n\", __func__, c, owner);\n"
                "                    ggml_cuda_ar_pipeline_free(p);\n"
                "                    return nullptr;\n"
                "                }\n"
                "                p->host_buf_dev3[c][owner] = reinterpret_cast<uint8_t *>(stage_alias);\n"
                "            }\n"
                "        }\n"
                "    }"
            ),
            guard=r"if \(n_devices == 3\) \{\n        for \(int c = 0; c < 3; \+\+c\)",
        ),
        Edit(
            id="relax-n-eq-2-assert",
            anchor=(
                r"    const int n = p->n_devices;\n"
                r"    GGML_ASSERT\(n == 2\);"
            ),
            rationale="allow ggml_cuda_ar_allreduce to proceed for n==3 as well",
            mode="replace",
            text=(
                "    const int n = p->n_devices;\n"
                "    GGML_ASSERT(n == 2 || n == 3);"
            ),
            guard=r"GGML_ASSERT\(n == 2 \|\| n == 3\);",
        ),
        Edit(
            id="dispatch-n3-early",
            anchor=r"    const size_t   input_nbytes = ggml_nbytes\(tensors\[0\]\);",
            rationale="self-contained early n==3 branch -- returns before any "
                      "N=2-only code (BF16 conversion, copy-engine path, the "
                      "pairwise chunked-kernel loop) is reached, so the existing "
                      "N=2 path is completely untouched",
            mode="insert_after",
            text=(
                "\n"
                "\n"
                "    if (n == 3) {\n"
                "        // GP11 preliminary: F32 only, chunked-kernel-eligible sizes\n"
                "        // only -- falls back to the caller's next provider (RCCL) for\n"
                "        // non-F32 types or large/prefill sizes, matching this\n"
                "        // project's own validated data (fused root beats RCCL at\n"
                "        // decode-realistic sizes but loses to it at large sizes).\n"
                "        if (input_type != GGML_TYPE_F32) {\n"
                "            return false;\n"
                "        }\n"
                "        if (p->copy_threshold > 0 && input_nbytes >= p->copy_threshold) {\n"
                "            return false;\n"
                "        }\n"
                "        return ggml_cuda_ar_allreduce_root3(p, backends, tensors, ne);\n"
                "    }"
            ),
            guard=r"if \(n == 3\) \{\n        // GP11 preliminary: F32 only",
        ),
    ),
)

GGML_CUDA_CU = FilePatch(
    path="ggml/src/ggml-cuda/ggml-cuda.cu",
    description="GP11 preliminary: relax the comm dispatcher's own hardcoded "
                "n_backends==2 gate so it can reach the (also relaxed) internal "
                "AllReduce pipeline for 3 devices",
    edits=(
        Edit(
            id="comm-dispatch-accept-n3",
            anchor=(
                r"    GGML_ASSERT\(comm_ctx->ar_pipeline != nullptr\);\n"
                r"\n"
                r"    const size_t n_backends = comm_ctx->backends.size\(\);\n"
                r"    GGML_ASSERT\(n_backends == 2\);"
            ),
            rationale="ggml_backend_cuda_comm_allreduce_internal's own gate, "
                      "separate from allreduce.cu's ggml_cuda_ar_allreduce -- "
                      "both had to be relaxed for the n==3 path to be reachable "
                      "at all (found via real hardware testing: the first build "
                      "compiled clean but crashed here at runtime, a genuine gap "
                      "the anchor-only dry-run pass didn't catch)",
            mode="replace",
            text=(
                "    GGML_ASSERT(comm_ctx->ar_pipeline != nullptr);\n"
                "\n"
                "    const size_t n_backends = comm_ctx->backends.size();\n"
                "    GGML_ASSERT(n_backends == 2 || n_backends == 3);"
            ),
            guard=r"GGML_ASSERT\(n_backends == 2 \|\| n_backends == 3\);",
        ),
    ),
)

PATCHES = [CU, GGML_CUDA_CU]
