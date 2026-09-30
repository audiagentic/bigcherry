"""1272: selectable compressed wire formats for the 2-GPU host AllReduce path.

This phase implements explicit f32|bf16|f16 wire overrides.  q8_0 is parsed
and fails closed until its block codec lands; an unset GGML_CUDA_AR_WIRE
falls through to the pristine b11233 BF16-threshold behaviour unchanged.
"""

import re as _re

from bigcherry.patcher import Edit, FilePatch

GROUP = "gpu-collectives"
STATE = "untested"

_INCLUDES_OLD = """#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <limits>
"""

_INCLUDES_NEW = """#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <type_traits>
"""

_EVENT_SLOT_ANCHOR = """struct ggml_cuda_ar_event_slot {
    cudaEvent_t app = nullptr;  // upstream computation complete
"""

_WIRE_SUPPORT = r'''enum class ggml_cuda_ar_wire_override {
    pristine,
    f32,
    bf16,
    f16,
    q8_0_unsupported,
};

static ggml_cuda_ar_wire_override ggml_cuda_ar_wire_from_env() {
    const char * value = getenv("GGML_CUDA_AR_WIRE");
    if (value == nullptr || value[0] == '\0') {
        return ggml_cuda_ar_wire_override::pristine;
    }
    if (strcmp(value, "f32") == 0) {
        return ggml_cuda_ar_wire_override::f32;
    }
    if (strcmp(value, "bf16") == 0) {
        return ggml_cuda_ar_wire_override::bf16;
    }
    if (strcmp(value, "f16") == 0) {
        return ggml_cuda_ar_wire_override::f16;
    }
    if (strcmp(value, "q8_0") == 0) {
        return ggml_cuda_ar_wire_override::q8_0_unsupported;
    }
    GGML_LOG_WARN("%s: unknown GGML_CUDA_AR_WIRE value '%s'; using pristine wire selection\n",
                  __func__, value);
    return ggml_cuda_ar_wire_override::pristine;
}

static const char * ggml_cuda_ar_wire_name(ggml_cuda_ar_wire_override wire) {
    switch (wire) {
        case ggml_cuda_ar_wire_override::f32:  return "f32";
        case ggml_cuda_ar_wire_override::bf16: return "bf16";
        case ggml_cuda_ar_wire_override::f16:  return "f16";
        case ggml_cuda_ar_wire_override::q8_0_unsupported: return "q8_0";
        case ggml_cuda_ar_wire_override::pristine: break;
    }
    return "pristine";
}

static void ggml_cuda_ar_trace_wire(ggml_cuda_ar_wire_override wire) {
    if (getenv("BIGCHERRY_PATCH_TRACE") == nullptr) {
        return;
    }
    static std::atomic_flag logged = ATOMIC_FLAG_INIT;
    if (!logged.test_and_set(std::memory_order_relaxed)) {
        GGML_LOG_INFO("BIGCHERRY_PATCH_HIT patch=1272_ar_wire path=ar_wire_%s\n",
                      ggml_cuda_ar_wire_name(wire));
    }
}

'''

_PIPELINE_FIELD_OLD = """    size_t   bf16_threshold; // tensors >= this size (bytes) are reduced via FP32->BF16 round-trip; 0 disables
    uint64_t call_count;
"""

_PIPELINE_FIELD_NEW = """    size_t   bf16_threshold; // tensors >= this size (bytes) are reduced via FP32->BF16 round-trip; 0 disables
    ggml_cuda_ar_wire_override wire_override;
    uint64_t call_count;
"""

_BF16_COMMENT_ANCHOR = """    // Default 1: BF16 round-trip is always on for F32 inputs (any non-zero
    // ne).  Set GGML_CUDA_AR_BF16_THRESHOLD=0 to disable, or to a larger
    // byte threshold to opt out for small tensors.
"""

_WIRE_INIT = '''    p->wire_override = ggml_cuda_ar_wire_from_env();
'''

_PIPELINE_COMMENT_ANCHOR = """// ---------------------------------------------------------------------------
// Pipeline structure
// ---------------------------------------------------------------------------
"""

_CONVERT_KERNEL = r'''template <typename T_src, typename T_wire>
static __global__ void ggml_cuda_ar_convert_kernel(
        const T_src * __restrict__ src,
        T_wire      * __restrict__ dst,
        int count) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int nt  = gridDim.x * blockDim.x;
    for (int i = tid; i < count; i += nt) {
        dst[i] = ggml_cuda_cast<T_wire>(src[i]);
    }
}

'''

_ALLREDUCE_ANCHOR = """bool ggml_cuda_ar_allreduce(
        ggml_cuda_ar_pipeline * p,
        ggml_backend_t        * backends,
        ggml_tensor           ** tensors) {
"""

_WIRE_HELPERS = r'''template <typename T_dst, typename T_wire>
static bool ggml_cuda_ar_allreduce_wire_typed(
        ggml_cuda_ar_pipeline * p,
        ggml_backend_t        * backends,
        ggml_tensor           ** tensors,
        const bool              compute_flag[GGML_CUDA_MAX_DEVICES],
        int64_t                 ne) {
    GGML_ASSERT(p->n_devices == 2);
    GGML_ASSERT(ne > 0);
    GGML_ASSERT(ne <= std::numeric_limits<int>::max());

    ggml_cuda_ar_trace_wire(p->wire_override);

    const size_t wire_size = sizeof(T_wire);
    GGML_ASSERT(p->buf_bytes >= wire_size);
    const size_t nbytes = (size_t) ne * wire_size;
    const bool use_copy_engine =
        p->copy_threshold > 0 &&
        nbytes >= p->copy_threshold;

    if (use_copy_engine) {
        if constexpr (std::is_same<T_dst, T_wire>::value) {
            T_dst * buf[GGML_CUDA_MAX_DEVICES] = {};
            for (int i = 0; i < p->n_devices; ++i) {
                buf[i] = static_cast<T_dst *>(tensors[i]->data);
            }
            return ggml_cuda_ar_allreduce_copy_outer<T_wire, T_dst>(
                p, backends, buf, buf, compute_flag, ne);
        } else {
            ggml_cuda_pool_alloc<T_wire> wire_tmp[GGML_CUDA_MAX_DEVICES];
            T_wire * src[GGML_CUDA_MAX_DEVICES] = {};
            T_dst  * dst[GGML_CUDA_MAX_DEVICES] = {};
            bool inner_compute[GGML_CUDA_MAX_DEVICES] = {};

            const int block_size = 256;
            int n_blocks = (int) ((ne + block_size - 1) / block_size);
            if (n_blocks > 1024) {
                n_blocks = 1024;
            }

            for (int i = 0; i < p->n_devices; ++i) {
                auto * cuda_ctx = static_cast<ggml_backend_cuda_context *>(backends[i]->context);
                GGML_ASSERT(cuda_ctx->device == p->devices[i]);
                ggml_cuda_set_device(p->devices[i]);

                wire_tmp[i].pool = &cuda_ctx->pool();
                wire_tmp[i].alloc(ne);
                src[i] = wire_tmp[i].get();
                dst[i] = static_cast<T_dst *>(tensors[i]->data);
                inner_compute[i] = true;

                if (compute_flag[i]) {
                    ggml_cuda_ar_convert_kernel<T_dst, T_wire><<<n_blocks, block_size, 0, cuda_ctx->stream()>>>(
                        dst[i], src[i], (int) ne);
                    CUDA_CHECK(cudaGetLastError());
                } else {
                    CUDA_CHECK(cudaMemsetAsync(src[i], 0, nbytes, cuda_ctx->stream()));
                    CUDA_CHECK(cudaMemsetAsync(dst[i], 0, (size_t) ne * sizeof(T_dst), cuda_ctx->stream()));
                }
            }

            return ggml_cuda_ar_allreduce_copy_outer<T_wire, T_dst>(
                p, backends, src, dst, inner_compute, ne);
        }
    }

    const size_t max_chunk_elems = p->buf_bytes / wire_size;
    GGML_ASSERT(max_chunk_elems > 0);

    for (int64_t chunk_start = 0; chunk_start < ne; chunk_start += (int64_t) max_chunk_elems) {
        const size_t remaining_elems = (size_t) (ne - chunk_start);
        const size_t chunk_elems = remaining_elems < max_chunk_elems ? remaining_elems : max_chunk_elems;
        const size_t chunk_dst_bytes = chunk_elems * sizeof(T_dst);

        const auto [slot, token] = ggml_cuda_ar_acquire_slot(p);
        const bool last_chunk = chunk_start + (int64_t) chunk_elems == ne;

        for (int i = 0; i < p->n_devices; ++i) {
            const int peer = 1 - i;
            ggml_cuda_set_device(p->devices[i]);
            auto * cuda_ctx = static_cast<ggml_backend_cuda_context *>(backends[i]->context);
            GGML_ASSERT(cuda_ctx->device == p->devices[i]);
            cudaStream_t stream = cuda_ctx->stream();

            T_dst * data = static_cast<T_dst *>(tensors[i]->data) + chunk_start;
            if (!compute_flag[i]) {
                CUDA_CHECK(cudaMemsetAsync(data, 0, chunk_dst_bytes, stream));
            }

            ggml_cuda_ar_kernel<T_dst, T_wire><<<dim3(GGML_CUDA_AR_KERNEL_BLOCKS), dim3(256), 0, stream>>>(
                data,
                data,
                reinterpret_cast<T_wire *>(p->host_buf[i].dev + (size_t) slot * p->buf_bytes),
                reinterpret_cast<const T_wire *>(p->host_buf[peer].dev + (size_t) slot * p->buf_bytes),
                static_cast<int>(chunk_elems),
                ggml_cuda_ar_arrival_ptr(p, slot, i),
                ggml_cuda_ar_arrival_ptr(p, slot, peer),
                token);
            CUDA_CHECK(cudaGetLastError());

            if (last_chunk) {
                CUDA_CHECK(cudaEventRecord(p->ev_pool[i][slot].ker, stream));
            }
        }
    }

    return true;
}

static bool ggml_cuda_ar_allreduce_wire_override(
        ggml_cuda_ar_pipeline * p,
        ggml_backend_t        * backends,
        ggml_tensor           ** tensors,
        const bool              compute_flag[GGML_CUDA_MAX_DEVICES],
        ggml_type               input_type,
        int64_t                 ne) {
    if (p->wire_override == ggml_cuda_ar_wire_override::q8_0_unsupported) {
        GGML_LOG_ERROR("%s: GGML_CUDA_AR_WIRE=q8_0 is not implemented by patch 1272 phase 1\n", __func__);
        return false;
    }

#define DISPATCH_AR_WIRE(T_dst) \
    switch (p->wire_override) { \
        case ggml_cuda_ar_wire_override::f32:  return ggml_cuda_ar_allreduce_wire_typed<T_dst, float>(p, backends, tensors, compute_flag, ne); \
        case ggml_cuda_ar_wire_override::bf16: return ggml_cuda_ar_allreduce_wire_typed<T_dst, nv_bfloat16>(p, backends, tensors, compute_flag, ne); \
        case ggml_cuda_ar_wire_override::f16:  return ggml_cuda_ar_allreduce_wire_typed<T_dst, half>(p, backends, tensors, compute_flag, ne); \
        case ggml_cuda_ar_wire_override::pristine: \
        case ggml_cuda_ar_wire_override::q8_0_unsupported: break; \
    }

    switch (input_type) {
        case GGML_TYPE_F32:  DISPATCH_AR_WIRE(float);
        case GGML_TYPE_F16:  DISPATCH_AR_WIRE(half);
        case GGML_TYPE_BF16: DISPATCH_AR_WIRE(nv_bfloat16);
        default: GGML_ASSERT(false);
    }

#undef DISPATCH_AR_WIRE
    return false;
}

'''

_COMPUTE_FLAGS_ANCHOR = """    bool compute_flag[GGML_CUDA_MAX_DEVICES] = {};
    for (int i = 0; i < n; ++i) {
        compute_flag[i] = (tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) != 0;
    }
"""

_OVERRIDE_DISPATCH = '''

    // Explicit wire overrides are a separate 2-GPU path.  With 1244 applied,
    // n==3 returns through root3 before reaching this pristine N=2 block.
    if (p->wire_override != ggml_cuda_ar_wire_override::pristine) {
        return ggml_cuda_ar_allreduce_wire_override(
            p, backends, tensors, compute_flag, input_type, ne);
    }
'''

PATCHES = [
    FilePatch(
        path="ggml/src/ggml-cuda/allreduce.cu",
        description="1272: explicit f32/bf16/f16 host AllReduce wire overrides on chunked and copy-engine paths",
        language="none",
        edits=(
            Edit(
                id="ar-wire-includes",
                anchor=_re.escape(_INCLUDES_OLD),
                mode="replace",
                text=_INCLUDES_NEW,
                guard=r"#include <atomic>",
                rationale="Add host-only atomic once-marker support and compile-time type equality for explicit wire dispatch.",
                expect_matches=1,
                max_span_lines=5,
            ),
            Edit(
                id="ar-wire-enum-parser-trace",
                anchor=_re.escape(_EVENT_SLOT_ANCHOR),
                mode="insert_before",
                text=_WIRE_SUPPORT,
                guard=r"enum class ggml_cuda_ar_wire_override",
                rationale="Attach the explicit wire selector and once-per-process BIGCHERRY_PATCH_TRACE marker immediately before the pristine event-slot declaration.",
                expect_matches=1,
                max_span_lines=2,
            ),
            Edit(
                id="ar-wire-pipeline-field",
                anchor=_re.escape(_PIPELINE_FIELD_OLD),
                mode="replace",
                text=_PIPELINE_FIELD_NEW,
                guard=r"ggml_cuda_ar_wire_override wire_override;",
                rationale="Persist the parsed wire override in the pipeline while leaving the pristine BF16 threshold field and behavior intact.",
                expect_matches=1,
                max_span_lines=2,
            ),
            Edit(
                id="ar-wire-init",
                anchor=_re.escape(_BF16_COMMENT_ANCHOR),
                mode="insert_before",
                text=_WIRE_INIT,
                guard=r"p->wire_override = ggml_cuda_ar_wire_from_env\(\);",
                rationale="Parse GGML_CUDA_AR_WIRE once at pipeline initialization; unset remains pristine before the existing BF16-threshold initialization.",
                expect_matches=1,
                max_span_lines=3,
            ),
            Edit(
                id="ar-wire-convert-kernel",
                anchor=_re.escape(_PIPELINE_COMMENT_ANCHOR),
                mode="insert_before",
                text=_CONVERT_KERNEL,
                guard=r"static __global__ void ggml_cuda_ar_convert_kernel\(",
                rationale="Provide device-side source-to-wire conversion used only when explicit copy-engine wire type differs from the tensor type.",
                expect_matches=1,
                max_span_lines=3,
            ),
            Edit(
                id="ar-wire-helpers",
                anchor=_re.escape(_ALLREDUCE_ANCHOR),
                mode="insert_before",
                text=_WIRE_HELPERS,
                guard=r"static bool ggml_cuda_ar_allreduce_wire_override\(",
                rationale="Add a separate explicit-wire implementation after copy_outer is defined, preserving the pristine allreduce body for the unset case and handling chunked/copy-engine paths with fused F32 accumulation.",
                expect_matches=1,
                max_span_lines=4,
            ),
            Edit(
                id="ar-wire-dispatch",
                anchor=_re.escape(_COMPUTE_FLAGS_ANCHOR),
                mode="insert_after",
                text=_OVERRIDE_DISPATCH,
                guard=r"return ggml_cuda_ar_allreduce_wire_override\(",
                rationale="Route only explicit wire selections into 1272 after compute flags are known; 1244's n==3 early return remains ahead of this site and untouched.",
                expect_matches=1,
                max_span_lines=4,
            ),
        ),
    ),
]
