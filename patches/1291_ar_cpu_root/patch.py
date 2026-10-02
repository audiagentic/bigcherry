"""1291: --allreduce cpu-root -- CPU-root one-shot AllReduce for small messages (HIP only).

Flash-Next on 2x 7900 XTX + R9700 (no P2P, RCCL over SHM) issues 96 decode AllReduces per token of
10 KB f32. Measured stream-ordered: RCCL 3 ranks 33 us/call, this protocol 13.8 us/call
(tools/lab/rccl/cpu-root-ar.hip). Each rank's kernel copies its slice into pinned, device-mapped host
memory and publishes a per-rank 128-B epoch line; one persistent CPU worker waits for every rank, sums
in fixed rank order (exact f32, deterministic) and publishes a result epoch; each rank's kernel spins on
that epoch and copies the result back. Everything is enqueued on each backend's own stream, so the host
thread never blocks. Slots and results are double-buffered by epoch parity; element counts use a ring.
Messages above BIGCHERRY_AR_CPU_ROOT_MAX_BYTES (default 65536), non-f32 or non-contiguous tensors go to
RCCL. BIGCHERRY_PATCH_HIT patch=1291_ar_cpu_root logs once under BIGCHERRY_PATCH_TRACE.
"""

import re as _re

from bigcherry.patcher import Edit, FilePatch

GROUP = "core"
STATE = "untested"

_FWD = """#if defined(GGML_USE_HIP)
// BigCherry 1291: CPU-root one-shot AllReduce state (defined with the provider below).
struct bc_cpu_root;
static void bc_cpu_root_free(bc_cpu_root * cr);
#endif // GGML_USE_HIP

"""

_IMPL = r'''
#if defined(GGML_USE_HIP)
// BigCherry 1291: CPU-root one-shot AllReduce for small f32 messages (see patches/1291_ar_cpu_root).
#include <thread>
#if defined(__x86_64__)
#include <immintrin.h>
#define BC_CPU_ROOT_PAUSE() _mm_pause()
#else
#define BC_CPU_ROOT_PAUSE() ((void) 0)
#endif

struct alignas(128) bc_cpu_root_line {
    volatile uint32_t v;      // epoch (arrive/done) or counter (ctr)
    volatile uint32_t n;      // element count published with an arrival
    char pad[120];
};

// Graph-safe: every per-call value is produced on the device at run time, never baked into kernel
// arguments (HIP graph capture/replay of decode and MTP verify would freeze a host-side epoch).
struct bc_cpu_root {
    int n = 0;
    size_t max_elems = 0;
    float * slots = nullptr;              // [n][2][max_elems], pinned + mapped
    float * result = nullptr;             // [2][max_elems]
    bc_cpu_root_line * arrive = nullptr;  // [n]  rank r: epoch + count of its latest slice
    bc_cpu_root_line * ctr = nullptr;     // [n]  rank r's own call counter (advanced by produce)
    bc_cpu_root_line * done = nullptr;    // [1]  CPU: epoch of the latest published result
    std::atomic<bool> stop{false};
    std::thread worker;
    bool traced = false;
};

static __global__ void bc_cpu_root_produce(const float * __restrict__ src, float * slots2,
        bc_cpu_root_line * ctr, bc_cpu_root_line * line, size_t max_elems, int n) {
    __shared__ uint32_t s_gen;
    if (threadIdx.x == 0) {
        s_gen = ((volatile bc_cpu_root_line *) ctr)->v + 1;
        ((volatile bc_cpu_root_line *) ctr)->v = s_gen;
    }
    __syncthreads();
    const uint32_t gen = s_gen;
    float * slot = slots2 + (size_t) (gen & 1) * max_elems;
    const int n4 = n / 4;
    const float4 * s4 = (const float4 *) src;
    float4 * d4 = (float4 *) slot;
    for (int i = threadIdx.x; i < n4; i += blockDim.x) {
        d4[i] = s4[i];
    }
    for (int i = n4 * 4 + threadIdx.x; i < n; i += blockDim.x) {
        slot[i] = src[i];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        line->n = (uint32_t) n;
        __threadfence_system();
        line->v = gen;
    }
}

static __global__ void bc_cpu_root_consume(float * __restrict__ dst, const float * result2,
        const bc_cpu_root_line * ctr, const bc_cpu_root_line * done, size_t max_elems, int n) {
    __shared__ uint32_t s_gen;
    if (threadIdx.x == 0) {
        const uint32_t gen = ((const volatile bc_cpu_root_line *) ctr)->v;  // set by this rank's produce
        while ((int32_t) (((const volatile bc_cpu_root_line *) done)->v - gen) < 0) {
            __builtin_amdgcn_s_sleep(1);
        }
        s_gen = gen;
    }
    __syncthreads();
    __threadfence_system();
    const float * res = result2 + (size_t) (s_gen & 1) * max_elems;
    const int n4 = n / 4;
    const float4 * r4 = (const float4 *) res;
    float4 * d4 = (float4 *) dst;
    for (int i = threadIdx.x; i < n4; i += blockDim.x) {
        d4[i] = r4[i];
    }
    for (int i = n4 * 4 + threadIdx.x; i < n; i += blockDim.x) {
        dst[i] = res[i];
    }
}

static void bc_cpu_root_worker(bc_cpu_root * cr) {
    uint32_t want = 1;
    while (!cr->stop.load(std::memory_order_relaxed)) {
        bool ready = true;
        for (int r = 0; r < cr->n; r++) {
            if ((int32_t) (cr->arrive[r].v - want) < 0) {
                ready = false;
                break;
            }
        }
        if (!ready) {
            BC_CPU_ROOT_PAUSE();
            continue;
        }
        std::atomic_thread_fence(std::memory_order_acquire);
        const size_t n = cr->arrive[0].n;
        const size_t p = want & 1;
        float * out = cr->result + p * cr->max_elems;
        const float * s0 = cr->slots + (0 * 2 + p) * cr->max_elems;
        for (size_t i = 0; i < n; i++) {
            out[i] = s0[i];
        }
        for (int r = 1; r < cr->n; r++) {  // fixed rank order: exact, deterministic f32
            const float * sr = cr->slots + ((size_t) r * 2 + p) * cr->max_elems;
            for (size_t i = 0; i < n; i++) {
                out[i] += sr[i];
            }
        }
        std::atomic_thread_fence(std::memory_order_release);
        cr->done->v = want;
        want++;
    }
}

static void bc_cpu_root_free(bc_cpu_root * cr) {
    if (cr == nullptr) {
        return;
    }
    cr->stop.store(true);
    if (cr->worker.joinable()) {
        cr->worker.join();
    }
    (void) hipHostFree(cr->slots);
    (void) hipHostFree(cr->result);
    (void) hipHostFree(cr->arrive);
    (void) hipHostFree(cr->done);
    (void) hipHostFree(cr->ctr);
    delete cr;
}

static bc_cpu_root * bc_cpu_root_create(int n, size_t max_bytes) {
    auto * cr = new bc_cpu_root;
    cr->n = n;
    cr->max_elems = (max_bytes / sizeof(float) + 3) & ~size_t(3);
    const unsigned int flags = hipHostMallocMapped | hipHostMallocPortable | hipHostMallocCoherent;
    bool ok = hipHostMalloc((void **) &cr->slots, sizeof(float) * cr->max_elems * 2 * n, flags) == hipSuccess;
    ok = ok && hipHostMalloc((void **) &cr->result, sizeof(float) * cr->max_elems * 2, flags) == hipSuccess;
    ok = ok && hipHostMalloc((void **) &cr->arrive, sizeof(bc_cpu_root_line) * n, flags) == hipSuccess;
    ok = ok && hipHostMalloc((void **) &cr->done, sizeof(bc_cpu_root_line), flags) == hipSuccess;
    ok = ok && hipHostMalloc((void **) &cr->ctr, sizeof(bc_cpu_root_line) * n, flags) == hipSuccess;
    if (!ok) {
        (void) hipGetLastError();
        bc_cpu_root_free(cr);
        return nullptr;
    }
    memset((void *) cr->arrive, 0, sizeof(bc_cpu_root_line) * n);
    memset((void *) cr->done, 0, sizeof(bc_cpu_root_line));
    memset((void *) cr->ctr, 0, sizeof(bc_cpu_root_line) * n);
    cr->worker = std::thread(bc_cpu_root_worker, cr);
    return cr;
}

static bool ggml_backend_cuda_comm_try_allreduce_cpu_root(
        ggml_backend_cuda_comm_context * comm_ctx, struct ggml_tensor ** tensors) {
    bc_cpu_root * cr = comm_ctx->cpu_root;
    const size_t n_ranks = comm_ctx->backends.size();
    bool small = cr != nullptr && tensors != nullptr && tensors[0] != nullptr &&
        tensors[0]->type == GGML_TYPE_F32 && (size_t) ggml_nelements(tensors[0]) <= cr->max_elems;
    for (size_t i = 0; small && i < n_ranks; i++) {
        small = tensors[i] != nullptr && ggml_is_contiguous(tensors[i]) &&
            ggml_nelements(tensors[i]) == ggml_nelements(tensors[0]) && tensors[i]->type == GGML_TYPE_F32;
    }
    if (!small) {
        comm_ctx->provider_name = "rccl";
#ifdef GGML_USE_NCCL
        if (!comm_ctx->comms.empty()) {
            return ggml_backend_cuda_comm_allreduce_nccl(comm_ctx, tensors);
        }
#endif // GGML_USE_NCCL
        return false;
    }
    comm_ctx->provider_name = "cpu-root";
    const int n = (int) ggml_nelements(tensors[0]);
    for (size_t i = 0; i < n_ranks; i++) {
        auto * cctx = static_cast<ggml_backend_cuda_context *>(comm_ctx->backends[i]->context);
        ggml_cuda_set_device(cctx->device);
        cudaStream_t stream = cctx->stream();
        float * data = (float *) tensors[i]->data;
        bc_cpu_root_produce<<<1, 1024, 0, stream>>>(data, cr->slots + i * 2 * cr->max_elems,
            cr->ctr + i, cr->arrive + i, cr->max_elems, n);
        bc_cpu_root_consume<<<1, 1024, 0, stream>>>(data, cr->result, cr->ctr + i, cr->done, cr->max_elems, n);
    }
    if (!cr->traced && getenv("BIGCHERRY_PATCH_TRACE") != nullptr) {
        cr->traced = true;
        GGML_LOG_WARN("BIGCHERRY_PATCH_HIT patch=1291_ar_cpu_root path=small ranks=%zu bytes=%zu\n",
            n_ranks, (size_t) n * sizeof(float));
    }
    return true;
}

static void ggml_backend_cuda_comm_init_cpu_root(ggml_backend_cuda_comm_context * ret) {
    ggml_backend_cuda_comm_init_nccl(ret);  // RCCL for messages above the small threshold
    const char * env = getenv("BIGCHERRY_AR_CPU_ROOT_MAX_BYTES");
    const size_t max_bytes = env != nullptr ? (size_t) strtoull(env, nullptr, 10) : 65536;
    ret->cpu_root = bc_cpu_root_create((int) ret->dev_ids.size(), max_bytes);
    if (ret->cpu_root == nullptr) {
        GGML_LOG_WARN("cpu-root: pinned host allocation failed; using the %s provider\n", ret->provider_name);
        return;
    }
    ret->try_allreduce = ggml_backend_cuda_comm_try_allreduce_cpu_root;
    ret->provider_name = "cpu-root";
}
#endif // GGML_USE_HIP
'''

CUDA = FilePatch(
    path="ggml/src/ggml-cuda/ggml-cuda.cu",
    language="none",
    description="--allreduce cpu-root: CPU-root one-shot AllReduce for small f32 messages, RCCL above.",
    edits=(
        Edit(
            id="cpu-root-forward-decl",
            anchor=_re.escape("struct ggml_backend_cuda_comm_context {\n"),
            text=_FWD,
            mode="insert_before",
            guard=r"BigCherry 1291: CPU-root one-shot AllReduce state",
            expect_matches=1,
            rationale="The comm context owns and frees the cpu-root state, defined later with the provider.",
        ),
        Edit(
            id="cpu-root-context-field",
            anchor=_re.escape("    ggml_cuda_ar_pipeline *     ar_pipeline = nullptr;\n"),
            text="#if defined(GGML_USE_HIP)\n    bc_cpu_root *               cpu_root = nullptr;  // BigCherry 1291\n#endif // GGML_USE_HIP\n",
            mode="insert_after",
            guard=r"cpu_root = nullptr;  // BigCherry 1291",
            expect_matches=1,
            rationale="Per-communicator cpu-root state, beside the internal pipeline.",
        ),
        Edit(
            id="cpu-root-context-free",
            anchor=_re.escape("        ggml_cuda_ar_pipeline_free(ar_pipeline);\n    }\n"),
            text=(
                "        ggml_cuda_ar_pipeline_free(ar_pipeline);\n"
                "#if defined(GGML_USE_HIP)\n"
                "        bc_cpu_root_free(cpu_root);  // BigCherry 1291\n"
                "#endif // GGML_USE_HIP\n"
                "    }\n"
            ),
            mode="replace",
            guard=r"bc_cpu_root_free\(cpu_root\);  // BigCherry 1291",
            expect_matches=1,
            rationale="Stop the worker and free the pinned buffers with the communicator.",
        ),
        Edit(
            id="cpu-root-known-provider",
            anchor=_re.escape('        p == "adaptive" || p == "p2p" || p == "root3" || p == "butterfly";\n'),
            text='        p == "adaptive" || p == "p2p" || p == "root3" || p == "butterfly" || p == "cpu-root";\n',
            mode="replace",
            guard=r'p == "butterfly" \|\| p == "cpu-root";',
            expect_matches=1,
            rationale="Accept --allreduce cpu-root in 0860's provider validation.",
        ),
        Edit(
            id="cpu-root-impl",
            anchor=_re.escape("static void * ggml_backend_cuda_comm_init(ggml_backend_t * backends, size_t n_backends) {\n"),
            text=_IMPL + "\n",
            mode="insert_before",
            guard=r"static bool ggml_backend_cuda_comm_try_allreduce_cpu_root\(",
            expect_matches=1,
            rationale="Provider implementation after the NCCL helpers and init chain it reuses.",
        ),
        Edit(
            id="cpu-root-init-dispatch",
            anchor=_re.escape('    } else if (provider == "adaptive") {\n        ggml_backend_cuda_comm_init_hybrid(ret);\n'),
            text=(
                '#if defined(GGML_USE_HIP)\n'
                '    } else if (provider == "cpu-root") {\n'
                '        ggml_backend_cuda_comm_init_cpu_root(ret);  // BigCherry 1291\n'
                '#endif // GGML_USE_HIP\n'
            ),
            mode="insert_after",
            guard=r"ggml_backend_cuda_comm_init_cpu_root\(ret\);  // BigCherry 1291",
            expect_matches=1,
            rationale="Select cpu-root beside 0840's adaptive provider.",
        ),
    ),
)

ARG = FilePatch(
    path="common/arg.cpp",
    language="none",
    description="List cpu-root in --allreduce help.",
    edits=(
        Edit(
            id="cpu-root-arg-help",
            anchor=_re.escape('"multi-GPU AllReduce provider: auto|ccl|host|adaptive|p2p|root3|butterfly (default: auto)"'),
            text='"multi-GPU AllReduce provider: auto|ccl|host|adaptive|p2p|root3|butterfly|cpu-root (default: auto)"',
            mode="replace",
            guard=r"butterfly\|cpu-root \(default: auto\)",
            expect_matches=1,
            rationale="Help text only; validation is in ggml_backend_comm_set_config.",
        ),
    ),
)

BENCH = FilePatch(
    path="tools/llama-bench/llama-bench.cpp",
    language="none",
    description="List cpu-root in llama-bench --allreduce help.",
    edits=(
        Edit(
            id="cpu-root-bench-help",
            anchor=_re.escape("--allreduce <auto|ccl|host|adaptive|p2p|root3|butterfly> (default: auto)"),
            text="--allreduce <auto|ccl|host|adaptive|p2p|root3|butterfly|cpu-root> (default: auto)",
            mode="replace",
            guard=r"butterfly\|cpu-root> \(default: auto\)",
            expect_matches=1,
            rationale="Help text only.",
        ),
    ),
)

PATCHES = [CUDA, ARG, BENCH]
