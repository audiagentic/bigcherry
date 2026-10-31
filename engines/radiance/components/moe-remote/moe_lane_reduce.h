#pragma once

// Lane route inverse, token reduce and publish kernels (pure move of moe_remote_gpu.h, RCS02).
#include "moe_lane_common.h"

namespace r3_moe_gpu {

__global__ void lane_route_inverse(const int32_t* sorted, int32_t* inv, uint32_t routes, uint32_t slots) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= routes) return;
    const int32_t flat = sorted[i];
    if (flat >= 0 && (uint32_t)flat < slots) inv[flat] = (int32_t)i;
}

/* One row a token with any helper route: the router-weighted sum of its helper-computed experts, slot
 * order, f32 accumulator, narrowed once to bf16 (moe_gather arithmetic over the slots this helper owns).
 * Tokens without a route are not written; the engine's merge reads only tokens it routed here. */
__global__ void lane_reduce_tokens(const uint16_t* down, const int32_t* inv,
                                   const uint16_t* ew, uint16_t* dst, uint32_t rows, uint32_t hidden,
                                   uint32_t poison, LaneShape shape) {
    using namespace r3::moe_shm;
    const uint32_t token = blockIdx.x;
    if (token >= rows) return;
    __shared__ int32_t row[kMaxTopK];
    __shared__ float w[kMaxTopK];
    __shared__ uint32_t any;
    if (threadIdx.x == 0) {
        any = 0;
        for (uint32_t slot = 0; slot < shape.top_k; ++slot) {
            const uint32_t flat = token * shape.top_k + slot;
            row[slot] = inv[flat];
            w[slot] = __uint_as_float((uint32_t)ew[flat] << 16);
            any |= row[slot] >= 0 ? 1u : 0u;
        }
    }
    __syncthreads();
    if (!any) return;
    uint16_t* out = dst + (uint64_t)token * hidden;
    for (uint32_t col = threadIdx.x; col < hidden; col += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t slot = 0; slot < shape.top_k; ++slot) {
            if (row[slot] < 0) continue;
            const uint64_t at = (uint64_t)(uint32_t)row[slot] * hidden + col;
            acc += w[slot] * __uint_as_float((uint32_t)down[at] << 16);
        }
        uint32_t u = __float_as_uint(acc);
        u += 0x7fffu + ((u >> 16) & 1u);
        out[col] = poison == 1u ? (uint16_t)0x447au : poison == 2u ? (uint16_t)0 : (uint16_t)(u >> 16);
    }
    /* Mapped host memory is fine-grained/uncached. Drain every writer wave's posted stores
     * before the later release flag, matching libr4d's no-P2P all-reduce host path. */
#if defined(__gfx1100__) || defined(__gfx1030__)
    asm volatile("s_waitcnt_vscnt null, 0x0" ::: "memory");   /* gfx11 spelling of the store drain */
#else
    asm volatile("s_wait_storecnt 0x0" ::: "memory");
#endif
}

/* lane_reduce_tokens, eight columns a thread: 16-byte loads of the expert rows and one 16-byte store to the mapped
 * result. Per column the arithmetic is the scalar kernel's (slot order, f32 accumulator, one bf16 rounding), so the
 * output is bit-identical; the 2-byte stores to uncached host memory ran at ~12 GB/s. */
__global__ void lane_reduce_tokens_v(const uint16_t* down, const int32_t* inv,
                                     const uint16_t* ew, uint16_t* dst, uint32_t rows, uint32_t hidden,
                                     uint32_t poison, LaneShape shape) {
    using namespace r3::moe_shm;
    const uint32_t token = blockIdx.x;
    if (token >= rows) return;
    __shared__ int32_t row[kMaxTopK];
    __shared__ float w[kMaxTopK];
    __shared__ uint32_t any;
    if (threadIdx.x == 0) {
        any = 0;
        for (uint32_t slot = 0; slot < shape.top_k; ++slot) {
            const uint32_t flat = token * shape.top_k + slot;
            row[slot] = inv[flat];
            w[slot] = __uint_as_float((uint32_t)ew[flat] << 16);
            any |= row[slot] >= 0 ? 1u : 0u;
        }
    }
    __syncthreads();
    if (!any) return;
    uint4* out = reinterpret_cast<uint4*>(dst + (uint64_t)token * hidden);
    for (uint32_t c8 = threadIdx.x; c8 < hidden / 8; c8 += blockDim.x) {
        float acc[8];
#pragma unroll
        for (uint32_t k = 0; k < 8; ++k) acc[k] = 0.0f;
        for (uint32_t slot = 0; slot < shape.top_k; ++slot) {
            if (row[slot] < 0) continue;
            const uint4 d = *reinterpret_cast<const uint4*>(down + (uint64_t)(uint32_t)row[slot] * hidden + (uint64_t)c8 * 8);
            const uint32_t dw[4] = {d.x, d.y, d.z, d.w};
#pragma unroll
            for (uint32_t k = 0; k < 4; ++k) {
                acc[2 * k] += w[slot] * __uint_as_float(dw[k] << 16);
                acc[2 * k + 1] += w[slot] * __uint_as_float(dw[k] & 0xffff0000u);
            }
        }
        uint32_t o[4];
#pragma unroll
        for (uint32_t k = 0; k < 4; ++k) {
            uint32_t h[2];
#pragma unroll
            for (uint32_t j = 0; j < 2; ++j) {
                uint32_t u = __float_as_uint(acc[2 * k + j]);
                u += 0x7fffu + ((u >> 16) & 1u);
                h[j] = poison == 1u ? 0x447au : poison == 2u ? 0u : (u >> 16);
            }
            o[k] = h[0] | (h[1] << 16);
        }
        out[c8] = make_uint4(o[0], o[1], o[2], o[3]);
    }
#if defined(__gfx1100__) || defined(__gfx1030__)
    asm volatile("s_waitcnt_vscnt null, 0x0" ::: "memory");
#else
    asm volatile("s_wait_storecnt 0x0" ::: "memory");
#endif
}

/* Every mapped result writer has drained before this kernel can run. Publish completion with a
 * system-scope release; the engine-side join waits with a system-scope acquire. A pre-queued wait released at burst end
 * (RAD20) is above the lane's publish limit (device memory, set by the helper while it parks): it publishes nothing. */
__global__ void lane_publish_done(r3::moe_shm::Request* q, uint32_t* done_sequence, uint32_t sequence,
                                  const uint32_t* publish_limit) {
    if (blockIdx.x || threadIdx.x || sequence > __atomic_load_n(publish_limit, __ATOMIC_RELAXED)) return;
    q->trace[4] = wall_clock64();   /* R3_MOE_WORKER_TRACE: this device's clock */
    lane_store_sys_rel(done_sequence, sequence);
}

}  // namespace r3_moe_gpu
