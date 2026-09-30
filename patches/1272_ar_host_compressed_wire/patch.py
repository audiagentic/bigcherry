"""1272: selectable compressed wire formats for the 2-GPU host AllReduce path.

Explicit f32|bf16|f16|q8_0 wire overrides apply to both the mapped-host
chunked path and the copy-engine path.  Unset GGML_CUDA_AR_WIRE falls through
to pristine b11233 BF16-threshold behaviour unchanged.
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
    q8_0,
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
        return ggml_cuda_ar_wire_override::q8_0;
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
        case ggml_cuda_ar_wire_override::q8_0: return "q8_0";
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

template <typename T_src>
static __global__ void ggml_cuda_ar_quantize_q8_0_kernel(
        const T_src * __restrict__ src,
        block_q8_0  * __restrict__ dst,
        int64_t ne,
        int64_t nblocks) {
    const int lane = threadIdx.x % QK8_0;
    const int64_t warp = ((int64_t) blockIdx.x * blockDim.x + threadIdx.x) / QK8_0;
    const int64_t nwarps = ((int64_t) gridDim.x * blockDim.x) / QK8_0;

    for (int64_t ib = warp; ib < nblocks; ib += nwarps) {
        const int64_t i = ib * QK8_0 + lane;
        const float x = i < ne ? ggml_cuda_cast<float>(src[i]) : 0.0f;
        const float amax = warp_reduce_max<QK8_0>(fabsf(x));
        const float d = amax / 127.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;

        dst[ib].qs[lane] = (int8_t) roundf(x * id);
        if (lane == 0) {
            dst[ib].d = ggml_cuda_cast<half>(d);
        }
    }
}

template <typename T_dst>
static __global__ void ggml_cuda_ar_q8_0_add_kernel(
        T_dst             * __restrict__ dst,
        const block_q8_0  * __restrict__ local,
        const block_q8_0  * __restrict__ peer,
        int count) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int nt  = gridDim.x * blockDim.x;
    for (int i = tid; i < count; i += nt) {
        const int ib = i / QK8_0;
        const int iq = i % QK8_0;
        const float a = ggml_cuda_cast<float>(local[ib].d) * (float) local[ib].qs[iq];
        const float b = ggml_cuda_cast<float>(peer[ib].d)  * (float) peer[ib].qs[iq];
        dst[i] = ggml_cuda_cast<T_dst>(a + b);
    }
}

template <typename T_dst>
static __global__ void ggml_cuda_ar_q8_0_mapped_kernel(
        const T_dst       *              sendbuf,
        T_dst             *              recvbuf,
        block_q8_0        * __restrict__ host_mine,
        const block_q8_0  * __restrict__ host_other,
        int                              count,
        int *                            arrival_mine,
        int *                            arrival_other,
        int                              token) {
    constexpr int ARRIVAL_INTS = (int) (GGML_CUDA_AR_ARRIVAL_STRIDE / sizeof(int));
    constexpr int WARPS_PER_BLOCK = 256 / QK8_0;

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int lane = tid % QK8_0;
    const int warp_in_block = tid / QK8_0;
    const int first_wire_block = bid * WARPS_PER_BLOCK + warp_in_block;
    const int wire_stride = gridDim.x * WARPS_PER_BLOCK;
    const int nblocks = (count + QK8_0 - 1) / QK8_0;

    for (int ib = first_wire_block; ib < nblocks; ib += wire_stride) {
        const int i = ib * QK8_0 + lane;
        const float x = i < count ? ggml_cuda_cast<float>(sendbuf[i]) : 0.0f;
        const float amax = warp_reduce_max<QK8_0>(fabsf(x));
        const float d = amax / 127.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;

        host_mine[ib].qs[lane] = (int8_t) roundf(x * id);
        if (lane == 0) {
            host_mine[ib].d = ggml_cuda_cast<half>(d);
        }
    }

    __threadfence_system();
    __syncthreads();

    if (tid == 0) {
        int       * my_slot    = arrival_mine  + bid * ARRIVAL_INTS;
        const int * other_slot = arrival_other + bid * ARRIVAL_INTS;
        ggml_cuda_ar_signal_set(my_slot, token);
        __threadfence_system();
        while (ggml_cuda_ar_signal_get(other_slot) != token) {
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

    for (int ib = first_wire_block; ib < nblocks; ib += wire_stride) {
        const int i = ib * QK8_0 + lane;
        if (i < count) {
            const float a = ggml_cuda_cast<float>(host_mine[ib].d) * (float) host_mine[ib].qs[lane];
            const float b = ggml_cuda_cast<float>(host_other[ib].d) * (float) host_other[ib].qs[lane];
            recvbuf[i] = ggml_cuda_cast<T_dst>(a + b);
        }
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

template <typename T_dst>
static bool ggml_cuda_ar_allreduce_copy_q8_impl(
        ggml_cuda_ar_pipeline * p,
        ggml_backend_t        * backends,
        block_q8_0 * const      src_buf[GGML_CUDA_MAX_DEVICES],
        T_dst * const            dst_buf[GGML_CUDA_MAX_DEVICES],
        int64_t                  ne,
        size_t                   nbytes) {
    GGML_ASSERT(p->n_devices == 2);
    GGML_ASSERT(nbytes <= p->copy_bytes);
    GGML_ASSERT(ne <= std::numeric_limits<int>::max());

    const size_t chunk_bytes = ggml_cuda_ar_chunk_bytes(p, nbytes);
    GGML_ASSERT(chunk_bytes > 0);
    const int slot = ggml_cuda_ar_acquire_slot(p).slot;
    const size_t copy_chunks = (nbytes + chunk_bytes - 1) / chunk_bytes;
    GGML_ASSERT(copy_chunks <= GGML_CUDA_AR_COPY_MAX_CHUNKS);

    ggml_backend_cuda_context * cuda_ctx[2] = {};

    for (int i = 0; i < 2; ++i) {
        ggml_cuda_set_device(p->devices[i]);
        cuda_ctx[i] = static_cast<ggml_backend_cuda_context *>(backends[i]->context);
        GGML_ASSERT(cuda_ctx[i]->device == p->devices[i]);
        ggml_cuda_ar_wait_for_compute(p, cuda_ctx[i], i, slot);

        if (p->host_large_read_done_valid) {
            const int peer = 1 - i;
            CUDA_CHECK(cudaStreamWaitEvent(p->streams[i], p->host_large_read_done[peer]));
        }

        for (size_t c = 0; c < copy_chunks; ++c) {
            const size_t offset = c * chunk_bytes;
            const size_t this_bytes = std::min(chunk_bytes, nbytes - offset);
            CUDA_CHECK(cudaMemcpyAsync(
                p->host_large[i].host + offset, reinterpret_cast<char *>(src_buf[i]) + offset, this_bytes,
                cudaMemcpyDeviceToHost, p->streams[i]));
            CUDA_CHECK(cudaEventRecord(p->ev_pool[i][slot].cpy[c], p->streams[i]));
        }
    }

    for (int i = 0; i < 2; ++i) {
        const int peer = 1 - i;
        ggml_cuda_set_device(p->devices[i]);

        if (p->dev_tmp_kernel_done_valid) {
            CUDA_CHECK(cudaStreamWaitEvent(p->streams[i], p->dev_tmp_kernel_done[i]));
        }

        for (size_t c = 0; c < copy_chunks; ++c) {
            const size_t offset = c * chunk_bytes;
            const size_t this_bytes = std::min(chunk_bytes, nbytes - offset);
            CUDA_CHECK(cudaStreamWaitEvent(p->streams[i], p->ev_pool[peer][slot].cpy[c]));
            CUDA_CHECK(cudaMemcpyAsync(
                p->dev_tmp[i] + offset, p->host_large[peer].host + offset, this_bytes,
                cudaMemcpyHostToDevice, p->streams[i]));
        }

        CUDA_CHECK(cudaEventRecord(p->host_large_read_done[i], p->streams[i]));
        CUDA_CHECK(cudaEventRecord(p->ev_pool[i][slot].h2d, p->streams[i]));
        CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx[i]->stream(), p->ev_pool[i][slot].h2d));

        const int block_size = 256;
        int n_blocks = (int) ((ne + block_size - 1) / block_size);
        if (n_blocks > 1024) {
            n_blocks = 1024;
        }
        ggml_cuda_ar_q8_0_add_kernel<T_dst><<<n_blocks, block_size, 0, cuda_ctx[i]->stream()>>>(
            dst_buf[i], src_buf[i], reinterpret_cast<const block_q8_0 *>(p->dev_tmp[i]), (int) ne);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaEventRecord(p->dev_tmp_kernel_done[i], cuda_ctx[i]->stream()));
        CUDA_CHECK(cudaEventRecord(p->ev_pool[i][slot].ker, cuda_ctx[i]->stream()));
    }

    p->host_large_read_done_valid = true;
    p->dev_tmp_kernel_done_valid = true;
    return true;
}

template <typename T_dst>
static bool ggml_cuda_ar_allreduce_copy_q8_outer(
        ggml_cuda_ar_pipeline * p,
        ggml_backend_t        * backends,
        block_q8_0 * const      src_buf[GGML_CUDA_MAX_DEVICES],
        T_dst * const            dst_buf[GGML_CUDA_MAX_DEVICES],
        int64_t                  ne) {
    const int64_t outer_max_blocks = (int64_t) (p->copy_bytes / sizeof(block_q8_0));
    GGML_ASSERT(outer_max_blocks > 0);
    const int64_t outer_max_elems = outer_max_blocks * QK8_0;

    bool ok = true;
    for (int64_t outer_start = 0; outer_start < ne && ok; outer_start += outer_max_elems) {
        const int64_t outer_ne = std::min(outer_max_elems, ne - outer_start);
        const int64_t outer_blocks = (outer_ne + QK8_0 - 1) / QK8_0;
        const size_t outer_nbytes = (size_t) outer_blocks * sizeof(block_q8_0);

        block_q8_0 * src[GGML_CUDA_MAX_DEVICES] = {};
        T_dst * dst[GGML_CUDA_MAX_DEVICES] = {};
        for (int i = 0; i < p->n_devices; ++i) {
            src[i] = src_buf[i] + outer_start / QK8_0;
            dst[i] = dst_buf[i] + outer_start;
        }
        ok = ggml_cuda_ar_allreduce_copy_q8_impl<T_dst>(
            p, backends, src, dst, outer_ne, outer_nbytes);
    }
    return ok;
}

template <typename T_dst>
static bool ggml_cuda_ar_allreduce_wire_q8(
        ggml_cuda_ar_pipeline * p,
        ggml_backend_t        * backends,
        ggml_tensor           ** tensors,
        const bool              compute_flag[GGML_CUDA_MAX_DEVICES],
        int64_t                 ne) {
    GGML_ASSERT(p->n_devices == 2);
    GGML_ASSERT(ne > 0);
    GGML_ASSERT(ne <= std::numeric_limits<int>::max());

    ggml_cuda_ar_trace_wire(p->wire_override);

    const int64_t q8_blocks = (ne + QK8_0 - 1) / QK8_0;
    const size_t q8_nbytes = (size_t) q8_blocks * sizeof(block_q8_0);
    const bool use_copy_engine =
        p->copy_threshold > 0 &&
        q8_nbytes >= p->copy_threshold;

    if (use_copy_engine) {
        ggml_cuda_pool_alloc<block_q8_0> q8_tmp[GGML_CUDA_MAX_DEVICES];
        block_q8_0 * src[GGML_CUDA_MAX_DEVICES] = {};
        T_dst * dst[GGML_CUDA_MAX_DEVICES] = {};

        const int block_size = 256;
        int n_blocks = (int) ((q8_blocks * QK8_0 + block_size - 1) / block_size);
        if (n_blocks > 1024) {
            n_blocks = 1024;
        }

        for (int i = 0; i < p->n_devices; ++i) {
            auto * cuda_ctx = static_cast<ggml_backend_cuda_context *>(backends[i]->context);
            GGML_ASSERT(cuda_ctx->device == p->devices[i]);
            ggml_cuda_set_device(p->devices[i]);

            q8_tmp[i].pool = &cuda_ctx->pool();
            q8_tmp[i].alloc(q8_blocks);
            src[i] = q8_tmp[i].get();
            dst[i] = static_cast<T_dst *>(tensors[i]->data);

            if (compute_flag[i]) {
                ggml_cuda_ar_quantize_q8_0_kernel<T_dst><<<n_blocks, block_size, 0, cuda_ctx->stream()>>>(
                    dst[i], src[i], ne, q8_blocks);
                CUDA_CHECK(cudaGetLastError());
            } else {
                CUDA_CHECK(cudaMemsetAsync(src[i], 0, q8_nbytes, cuda_ctx->stream()));
                CUDA_CHECK(cudaMemsetAsync(dst[i], 0, (size_t) ne * sizeof(T_dst), cuda_ctx->stream()));
            }
        }

        return ggml_cuda_ar_allreduce_copy_q8_outer<T_dst>(p, backends, src, dst, ne);
    }

    const size_t max_wire_blocks = p->buf_bytes / sizeof(block_q8_0);
    GGML_ASSERT(max_wire_blocks > 0);
    const size_t max_chunk_elems = max_wire_blocks * QK8_0;

    for (int64_t chunk_start = 0; chunk_start < ne; chunk_start += (int64_t) max_chunk_elems) {
        const size_t remaining = (size_t) (ne - chunk_start);
        const size_t chunk_elems = std::min(max_chunk_elems, remaining);
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

            ggml_cuda_ar_q8_0_mapped_kernel<T_dst><<<dim3(GGML_CUDA_AR_KERNEL_BLOCKS), dim3(256), 0, stream>>>(
                data,
                data,
                reinterpret_cast<block_q8_0 *>(p->host_buf[i].dev + (size_t) slot * p->buf_bytes),
                reinterpret_cast<const block_q8_0 *>(p->host_buf[peer].dev + (size_t) slot * p->buf_bytes),
                (int) chunk_elems,
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
#define DISPATCH_AR_WIRE(T_dst) \
    switch (p->wire_override) { \
        case ggml_cuda_ar_wire_override::f32:  return ggml_cuda_ar_allreduce_wire_typed<T_dst, float>(p, backends, tensors, compute_flag, ne); \
        case ggml_cuda_ar_wire_override::bf16: return ggml_cuda_ar_allreduce_wire_typed<T_dst, nv_bfloat16>(p, backends, tensors, compute_flag, ne); \
        case ggml_cuda_ar_wire_override::f16:  return ggml_cuda_ar_allreduce_wire_typed<T_dst, half>(p, backends, tensors, compute_flag, ne); \
        case ggml_cuda_ar_wire_override::q8_0: return ggml_cuda_ar_allreduce_wire_q8<T_dst>(p, backends, tensors, compute_flag, ne); \
        case ggml_cuda_ar_wire_override::pristine: break; \
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
        description="1272: explicit f32/bf16/f16/q8_0 host AllReduce wire overrides on chunked and copy-engine paths",
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
                max_span_lines=3,
            ),
            Edit(
                id="ar-wire-pipeline-field",
                anchor=_re.escape(_PIPELINE_FIELD_OLD),
                mode="replace",
                text=_PIPELINE_FIELD_NEW,
                guard=r"ggml_cuda_ar_wire_override wire_override;",
                rationale="Persist the parsed wire override in the pipeline while leaving the pristine BF16 threshold field and behavior intact.",
                expect_matches=1,
                max_span_lines=3,
            ),
            Edit(
                id="ar-wire-init",
                anchor=_re.escape(_BF16_COMMENT_ANCHOR),
                mode="insert_before",
                text=_WIRE_INIT,
                guard=r"p->wire_override = ggml_cuda_ar_wire_from_env\(\);",
                rationale="Parse GGML_CUDA_AR_WIRE once at pipeline initialization; unset remains pristine before the existing BF16-threshold initialization.",
                expect_matches=1,
                max_span_lines=4,
            ),
            Edit(
                id="ar-wire-convert-kernel",
                anchor=_re.escape(_PIPELINE_COMMENT_ANCHOR),
                mode="insert_before",
                text=_CONVERT_KERNEL,
                guard=r"static __global__ void ggml_cuda_ar_q8_0_mapped_kernel\(",
                rationale="Provide explicit-wire conversion plus Q8_0 block-32/fp16-scale quantize, fused-dequant-add, and mapped-host kernels before pipeline declarations.",
                expect_matches=1,
                max_span_lines=4,
            ),
            Edit(
                id="ar-wire-helpers",
                anchor=_re.escape(_ALLREDUCE_ANCHOR),
                mode="insert_before",
                text=_WIRE_HELPERS,
                guard=r"static bool ggml_cuda_ar_allreduce_wire_override\(",
                rationale="Add explicit-wire dispatch after copy_outer is defined; Q8_0 has provider-preserving copy-engine and mapped-host paths and all receivers accumulate dequantized values in F32.",
                expect_matches=1,
                max_span_lines=5,
            ),
            Edit(
                id="ar-wire-dispatch",
                anchor=_re.escape(_COMPUTE_FLAGS_ANCHOR),
                mode="insert_after",
                text=_OVERRIDE_DISPATCH,
                guard=r"return ggml_cuda_ar_allreduce_wire_override\(",
                rationale="Route only explicit wire selections into 1272 after compute flags are known; 1244's n==3 early return remains ahead of this site and untouched.",
                expect_matches=1,
                max_span_lines=5,
            ),
        ),
    ),
]
