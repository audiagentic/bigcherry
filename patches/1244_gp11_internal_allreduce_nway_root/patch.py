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
// N=3 specialized fused reduce-to-root + broadcast (GP11).
//
// Rank 0 is always root. F32 only. Root and leaf are separate kernels
// (gpt-dev-agent design, req_a60b24c16aa542c1, applying the root3-
// specialized design validated in tools/lab/gp10-collective-harness/ to
// production) so the hot path contains no is_root/n_devices/root branching
// or peer-table lookup -- real hardware: 65us median at 30720 F32 elements
// vs the prior generic kernel's 68.3-68.8us, beats RCCL's 87.8us by ~27%.
//
// IMPORTANT correctness invariants (do not regress either):
//   * arrival_* arguments are BASE (slot,rank) pointers; each kernel adds
//     blockIdx.x * ARRIVAL_INTS itself (2026-09-02 fix, req_1f050a8abef749a2
//     -- the original generic kernel signalled the base pointer directly,
//     so all blocks shared one token: fast, no crash, garbage output).
//   * all mapped host pointers are consumer-specific aliases resolved by
//     host_buf_dev3[consumer][owner]/arrival_dev3[consumer] at init (the
//     same class of cross-device pointer-alias bug already fixed once in
//     the harness, reintroduced and re-fixed here).
//   * send/recv are NOT __restrict__: production AllReduce is in-place and
//     passes the same tensor pointer for both.
//   * UNROLL=1 only -- real hardware showed UNROLL=2/4 regress at this size
//     (not enough work per thread for the extra register pressure to pay
//     off), so the root's reduce loop is deliberately NOT unrolled.
// ---------------------------------------------------------------------------
static __global__ void ggml_cuda_ar_kernel3(
        const float4 *              send,
        float4       *              recv,
        float4       * __restrict__ publish,
        const float4 * __restrict__ leaf0,
        const float4 * __restrict__ leaf1,
        int                          count,
        int *                        leaf0_ready,
        int *                        leaf1_ready,
        int *                        root_ready,
        int                          token) {
    constexpr int ARRIVAL_INTS = (int) (GGML_CUDA_AR_ARRIVAL_STRIDE / sizeof(int));

    const int tid    = threadIdx.x;
    const int bid     = blockIdx.x;
    const int gtid    = bid * blockDim.x + tid;
    const int gnt     = gridDim.x * blockDim.x;
    const int count4  = count / 4;
    const int tail    = count4 * 4;

    const int * leaf0_slot = leaf0_ready + bid * ARRIVAL_INTS;
    const int * leaf1_slot = leaf1_ready + bid * ARRIVAL_INTS;
    int       * root_slot  = root_ready  + bid * ARRIVAL_INTS;

    // P1+P2: separate lanes poll each leaf (matches the stress-validated
    // harness exactly; a true cross-wave variant using tid==0/tid==warpSize
    // is a follow-up experiment, not yet validated for production).
    if (tid == 0) {
        while (ggml_cuda_ar_signal_get(leaf0_slot) != token) {
#ifdef GGML_USE_HIP
            __builtin_amdgcn_s_sleep(4);
#elif __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
            __nanosleep(100);
#else
            NO_DEVICE_CODE;
#endif
        }
    }
    if (tid == 1) {
        while (ggml_cuda_ar_signal_get(leaf1_slot) != token) {
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

    for (int i = gtid; i < count4; i += gnt) {
        // Explicit independent loads -- the material optimization that made
        // the N=3 root path beat RCCL (both mapped-host reads outstanding
        // simultaneously instead of a serial dependent-add).
        const float4 a = leaf0[i];
        const float4 b = leaf1[i];
        const float4 s = send[i];
        float4 out;
        out.x = s.x + a.x + b.x;
        out.y = s.y + a.y + b.y;
        out.z = s.z + a.z + b.z;
        out.w = s.w + a.w + b.w;
        recv[i]    = out;
        publish[i] = out;
    }

    // Production must retain arbitrary-F32-count correctness (the harness's
    // root3 mode required elements%4==0; production cannot assume that).
    // Only block 0 owns the scalar tail.
    if (bid == 0 && tid < count - tail) {
        const float * send_f    = reinterpret_cast<const float *>(send);
        float       * recv_f    = reinterpret_cast<float *>(recv);
        float       * publish_f = reinterpret_cast<float *>(publish);
        const float * leaf0_f   = reinterpret_cast<const float *>(leaf0);
        const float * leaf1_f   = reinterpret_cast<const float *>(leaf1);

        const int i = tail + tid;
        const float out = send_f[i] + leaf0_f[i] + leaf1_f[i];
        recv_f[i]    = out;
        publish_f[i] = out;
    }

    __threadfence_system();
    __syncthreads();
    if (tid == 0) {
        ggml_cuda_ar_signal_set(root_slot, token);
        __threadfence_system();
    }
}

static __global__ void ggml_cuda_ar_kernel3_leaf(
        const float4 *              send,
        float4       *              recv,
        float4       * __restrict__ mine,
        const float4 * __restrict__ root_stage,
        int                          count,
        int *                        mine_ready,
        int *                        root_ready,
        int                          token) {
    constexpr int ARRIVAL_INTS = (int) (GGML_CUDA_AR_ARRIVAL_STRIDE / sizeof(int));

    const int tid    = threadIdx.x;
    const int bid     = blockIdx.x;
    const int gtid    = bid * blockDim.x + tid;
    const int gnt     = gridDim.x * blockDim.x;
    const int count4  = count / 4;
    const int tail    = count4 * 4;

    int       * mine_slot = mine_ready + bid * ARRIVAL_INTS;
    const int * root_slot = root_ready + bid * ARRIVAL_INTS;

    for (int i = gtid; i < count4; i += gnt) {
        mine[i] = send[i];
    }
    if (bid == 0 && tid < count - tail) {
        const float * send_f = reinterpret_cast<const float *>(send);
        float       * mine_f = reinterpret_cast<float *>(mine);
        mine_f[tail + tid] = send_f[tail + tid];
    }
    __threadfence_system();
    __syncthreads();
    if (tid == 0) {
        ggml_cuda_ar_signal_set(mine_slot, token);
        __threadfence_system();
    }

    if (tid == 0) {
        while (ggml_cuda_ar_signal_get(root_slot) != token) {
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

    for (int i = gtid; i < count4; i += gnt) {
        recv[i] = root_stage[i];
    }
    if (bid == 0 && tid < count - tail) {
        float       * recv_f = reinterpret_cast<float *>(recv);
        const float * root_f = reinterpret_cast<const float *>(root_stage);
        recv_f[tail + tid] = root_f[tail + tid];
    }
}
'''

_ROOT3_HELPER_TEXT = '''
// N=3 arrival BASE pointer for a specific consuming GPU. The kernel adds its
// own per-block offset. Never substitute p->arrival.dev here: the N=3 path
// requires the alias resolved with the consuming device current.
static int * ggml_cuda_ar_arrival_ptr3(
        const ggml_cuda_ar_pipeline * p, int consumer, int slot, int rank) {
    const size_t offset = ((size_t) slot * 3 + rank) *
                          GGML_CUDA_AR_KERNEL_BLOCKS * GGML_CUDA_AR_ARRIVAL_STRIDE;
    return reinterpret_cast<int *>(
        reinterpret_cast<uint8_t *>(p->arrival_dev3[consumer]) + offset);
}

// N=3 specialized fused-root dispatcher (GP11, gpt-dev-agent design,
// req_a60b24c16aa542c1). Rank 0 is root; ranks 1/2 are leaves. F32-only
// admission and large-message RCCL fallback are enforced by the caller
// (ggml_cuda_ar_allreduce's dispatch-n3-early branch) before entering here.
static bool ggml_cuda_ar_allreduce_root3(
        ggml_cuda_ar_pipeline * p,
        ggml_backend_t        * backends,
        ggml_tensor           ** tensors,
        int64_t                 ne) {
    GGML_ASSERT(p->n_devices == 3);
    GGML_ASSERT(ne > 0);

    const size_t max_chunk_elems = p->buf_bytes / sizeof(float);
    GGML_ASSERT(max_chunk_elems > 0);
    // p->buf_bytes is 1 MiB today, hence every non-first chunk starts on a
    // float4 boundary. Keep this invariant explicit because the specialized
    // kernels use float4 tensor pointers directly.
    GGML_ASSERT((max_chunk_elems % 4) == 0);

    ggml_backend_cuda_context * cuda_ctx[3] = {};
    cudaStream_t streams[3] = {};
    float * data_base[3] = {};
    bool compute[3] = {};

    // Resolve call-invariant state once, outside the chunk loop.
    for (int i = 0; i < 3; ++i) {
        cuda_ctx[i] = static_cast<ggml_backend_cuda_context *>(backends[i]->context);
        GGML_ASSERT(cuda_ctx[i]->device == p->devices[i]);

        streams[i]   = cuda_ctx[i]->stream();
        data_base[i] = static_cast<float *>(tensors[i]->data);
        compute[i]   = (tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) != 0;

        // Inactive shards contribute zero. N=3 is F32-only, so zero the
        // complete inactive tensor once rather than issuing one memset per
        // chunk. Same-stream ordering guarantees all later root3 kernels
        // observe the zeroed contribution.
        if (!compute[i]) {
            ggml_cuda_set_device(p->devices[i]);
            CUDA_CHECK(cudaMemsetAsync(data_base[i], 0, (size_t) ne * sizeof(float), streams[i]));
        }
    }

    for (int64_t chunk_start = 0; chunk_start < ne; chunk_start += (int64_t) max_chunk_elems) {
        const size_t remaining = (size_t) (ne - chunk_start);
        const size_t chunk_elems = remaining < max_chunk_elems ? remaining : max_chunk_elems;
        const int chunk_count = static_cast<int>(chunk_elems);

        const auto [slot, token] = ggml_cuda_ar_acquire_slot(p);
        const size_t slot_offset = (size_t) slot * p->buf_bytes;

        float * data0 = data_base[0] + chunk_start;
        float * data1 = data_base[1] + chunk_start;
        float * data2 = data_base[2] + chunk_start;

        // -- Root = rank 0. Every mapped pointer below is from consumer 0's
        // alias table. --
        ggml_cuda_set_device(p->devices[0]);
        auto * root_publish = reinterpret_cast<float4 *>(p->host_buf_dev3[0][0] + slot_offset);
        auto * root_leaf0   = reinterpret_cast<const float4 *>(p->host_buf_dev3[0][1] + slot_offset);
        auto * root_leaf1   = reinterpret_cast<const float4 *>(p->host_buf_dev3[0][2] + slot_offset);
        ggml_cuda_ar_kernel3<<<dim3(GGML_CUDA_AR_KERNEL_BLOCKS), dim3(256), 0, streams[0]>>>(
            reinterpret_cast<const float4 *>(data0), reinterpret_cast<float4 *>(data0),
            root_publish, root_leaf0, root_leaf1, chunk_count,
            ggml_cuda_ar_arrival_ptr3(p, 0, slot, 1),
            ggml_cuda_ar_arrival_ptr3(p, 0, slot, 2),
            ggml_cuda_ar_arrival_ptr3(p, 0, slot, 0),
            token);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(p->ev_pool[0][slot].ker, streams[0]));

        // -- Leaf = rank 1. Every mapped pointer below is from consumer 1's
        // alias table. --
        ggml_cuda_set_device(p->devices[1]);
        auto * leaf1_mine = reinterpret_cast<float4 *>(p->host_buf_dev3[1][1] + slot_offset);
        auto * leaf1_root = reinterpret_cast<const float4 *>(p->host_buf_dev3[1][0] + slot_offset);
        ggml_cuda_ar_kernel3_leaf<<<dim3(GGML_CUDA_AR_KERNEL_BLOCKS), dim3(256), 0, streams[1]>>>(
            reinterpret_cast<const float4 *>(data1), reinterpret_cast<float4 *>(data1),
            leaf1_mine, leaf1_root, chunk_count,
            ggml_cuda_ar_arrival_ptr3(p, 1, slot, 1),
            ggml_cuda_ar_arrival_ptr3(p, 1, slot, 0),
            token);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(p->ev_pool[1][slot].ker, streams[1]));

        // -- Leaf = rank 2. Every mapped pointer below is from consumer 2's
        // alias table. --
        ggml_cuda_set_device(p->devices[2]);
        auto * leaf2_mine = reinterpret_cast<float4 *>(p->host_buf_dev3[2][2] + slot_offset);
        auto * leaf2_root = reinterpret_cast<const float4 *>(p->host_buf_dev3[2][0] + slot_offset);
        ggml_cuda_ar_kernel3_leaf<<<dim3(GGML_CUDA_AR_KERNEL_BLOCKS), dim3(256), 0, streams[2]>>>(
            reinterpret_cast<const float4 *>(data2), reinterpret_cast<float4 *>(data2),
            leaf2_mine, leaf2_root, chunk_count,
            ggml_cuda_ar_arrival_ptr3(p, 2, slot, 2),
            ggml_cuda_ar_arrival_ptr3(p, 2, slot, 0),
            token);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(p->ev_pool[2][slot].ker, streams[2]));
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
