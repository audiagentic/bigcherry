#pragma once

// Gated activation and E4M3 requantise between the large lane's gate/up and down GEMMs (RAD23). The plugin's
// gated_quant_fp8 / gated_had_quant_fp8 are launched with M = the lane's route capacity (rows x top_k), since the
// kernels are queued before a request says how many routes it has; on a helper that owns few routes nearly all of
// those rows are never read. This is the same kernel (libr4d r4d_fused_quant_fp8.hip, gated_quant_fp8_kernel<0, ROT>,
// y absent): one block a routed row, one wave a 128-wide group, four values a lane, the bf16-rounded SiLU product,
// the in-register Hadamard (ROT), the wave absmax and the packed E4M3 codes, from libr4d's own r4d_q8.h and
// r4d_fwht.h; a block past the request's route count returns. The E4M3 pack is the instruction on gfx12 and, where
// it does not exist, libr3's software conversion with the same rounding (libr3/r3_compat.h, r3_f32_to_e4m3).
#include "moe_lane_common.h"

namespace r3_moe_gpu {
__device__ __forceinline__ uint32_t lane_f32_to_e4m3(float f) {
    const uint32_t u = __float_as_uint(f);
    const uint32_t s = (u >> 24) & 0x80u;
    const uint32_t a = u & 0x7fffffffu;
    if (a > 0x7f800000u) return s | 0x7fu;
    if (a >= 0x43e00000u) return s | 0x7eu;  // >= 448
    const int e = static_cast<int>(a >> 23) - 127;
    uint32_t m = a & 0x7fffffu;
    if (e < -10) return s;
    if (e < -6) {  // subnormal code
        const uint32_t sh = static_cast<uint32_t>(-6 - e + 20);
        m |= 0x800000u;
        const uint32_t lo = m & ((1u << sh) - 1u);
        uint32_t q = m >> sh;
        const uint32_t half = 1u << (sh - 1u);
        if (lo > half || (lo == half && (q & 1u))) ++q;
        return s | q;
    }
    uint32_t q = m >> 20;
    const uint32_t lo = m & 0xfffffu;
    if (lo > 0x80000u || (lo == 0x80000u && (q & 1u))) ++q;
    int ee = e + 7;
    if (q == 0x8u) { q = 0; ++ee; }
    if (ee > 15 || (ee == 15 && q >= 7u)) return s | 0x7eu;
    return s | (static_cast<uint32_t>(ee) << 3) | q;
}

__device__ __forceinline__ int lane_cvt_pk_fp8_f32(float x, float y, int old, bool hi) {
    const uint32_t pair = lane_f32_to_e4m3(x) | (lane_f32_to_e4m3(y) << 8);
    const uint32_t keep = static_cast<uint32_t>(old);
    return static_cast<int>(hi ? ((keep & 0x0000ffffu) | (pair << 16)) : ((keep & 0xffff0000u) | pair));
}
}  // namespace r3_moe_gpu

#if defined(__HIP_DEVICE_COMPILE__) && !defined(__GFX12__)
#define __builtin_amdgcn_cvt_pk_fp8_f32 r3_moe_gpu::lane_cvt_pk_fp8_f32
#endif
#include "r4d_q8.h"
#include "r4d_fwht.h"
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__GFX12__)
#undef __builtin_amdgcn_cvt_pk_fp8_f32
#endif

namespace r3_moe_gpu {

constexpr uint32_t kQuantGroup = 128;

__device__ __forceinline__ float lane_bf2f(uint16_t h) { return __uint_as_float((uint32_t)h << 16); }
__device__ __forceinline__ uint16_t lane_f2bf(float f) {
    const uint32_t u = __float_as_uint(f);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
__device__ __forceinline__ float lane_wave_max(float v) {
    v = fmaxf(v, r4d_xor_lane<16>(v));
    v = fmaxf(v, r4d_xor_lane<8>(v));
    v = fmaxf(v, r4d_xor_lane<4>(v));
    v = fmaxf(v, r4d_xor_lane<2>(v));
    v = fmaxf(v, r4d_xor_lane<1>(v));
    return v;
}

template <bool ROT>
__global__ __launch_bounds__(1024) void lane_gated_quant(
    const r3::moe_shm::Request* q, uint32_t rank, uint32_t sequence, uint32_t max_rows, uint32_t layers, LaneShape shape,
    const uint16_t* __restrict__ gu, uint8_t* __restrict__ out, float* __restrict__ s,
    uint32_t n, uint32_t nkb, uint32_t gu_ld, uint32_t q_ld, uint32_t s_ld) {
    const uint32_t row = blockIdx.x;
    if (!lane_request_valid(q, rank, sequence, max_rows, layers, shape) || row >= q->route_count) return;
    const uint32_t lane = threadIdx.x & 31u, wave = threadIdx.x / 32u;
    const uint16_t* __restrict__ gr = gu + (size_t)row * gu_ld;
    uint8_t* __restrict__ qr = out + (size_t)row * q_ld;
    for (uint32_t kb = wave; kb < nkb; kb += blockDim.x / 32u) {
        const uint32_t c0 = kb * kQuantGroup + lane * 4u;
        float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        if (c0 < n) {
            uint16_t gb[4], ub[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) { gb[j] = gr[c0 + j]; ub[j] = gr[n + c0 + j]; }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float g = lane_bf2f(gb[j]);
                const float a = g / (1.0f + expf(-g));
                v[j] = lane_bf2f(lane_f2bf(a * lane_bf2f(ub[j])));
            }
        }
        if constexpr (ROT) r4d_fwht_wave<(int)kQuantGroup>(v, (int)lane);
        float m = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
        m = lane_wave_max(m);
        const float sc = m > 0.0f ? m * (1.0f / q8_top<false>()) : 1.0f;
        if (lane == 0) s[(size_t)row * s_ld + kb] = sc;
        const float inv = 1.0f / sc;
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__GFX12__)
        int pk = 0;
        pk = lane_cvt_pk_fp8_f32(fminf(fmaxf(v[0] * inv, -448.0f), 448.0f), fminf(fmaxf(v[1] * inv, -448.0f), 448.0f), pk, false);
        pk = lane_cvt_pk_fp8_f32(fminf(fmaxf(v[2] * inv, -448.0f), 448.0f), fminf(fmaxf(v[3] * inv, -448.0f), 448.0f), pk, true);
        const uint32_t pkd = (uint32_t)pk;
#else
        const uint32_t pkd = q8_pack4<false>(v[0], v[1], v[2], v[3], inv);
#endif
        if (c0 < n) *(uint32_t*)(qr + c0) = pkd;
    }
}

}  // namespace r3_moe_gpu
