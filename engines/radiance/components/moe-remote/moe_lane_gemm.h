#pragma once

// Lane expert GEMV (decode) and few-rows GEMM kernels (pure move of moe_remote_gpu.h, RCS02).
#include "moe_lane_common.h"

namespace r3_moe_gpu {

/* DECODE-SIZED MoE GEMV for w4nl64a8h (RAD05), the helper's small lane on gfx1100. The grouped GEMM's inner
 * loop is gfx12 fp8 WMMA, which gfx11 runs widened to bf16 -- ~4x off the weight-bandwidth bound at a decode
 * step's handful of rows a expert. Here no WMMA: one block a (routed row, 64 outputs), the row's activation
 * dequantised into LDS once, each wave a dot product per output straight off the packed nibbles.
 *
 * The weight is the relayout's own form (r4d_layout.cpp): a row of K codes is K/2 bytes, and inside each dword
 * byte b holds element b (low nibble) and element b+4 (high nibble) -- moew4_permute's order. A code indexes
 * kW4nlTable; its scale is one E4M3 byte a (row, 64 of K) times 2^-13 (kW4G64Scale). The activation is E4M3
 * with an f32 scale a 128 of K; the Hadamard rotation is already in both. f32 accumulate, one bf16 narrow.
 * TOKEN: gate/up, whose activation row is the routed slot's token (sorted[r] / top_k); otherwise down, whose
 * activation row is the sorted row itself. Rows [off[e], off[e+1]) belong to expert slot e; class e & 1, table
 * entry e / 2, and that class's N and K (the two TP halves of a two-rank engine differ). */
__device__ __forceinline__ float gemv_e4m3(uint8_t v) {
    const uint32_t s = v >> 7, e = (v >> 3) & 15u, m = v & 7u;
    float f = e ? __uint_as_float(((e + 120u) << 23) | (m << 20)) : (float)m * 0x1p-9f;
    return s ? -f : f;
}

/* E4M3 to f32 without a branch (RAD14): the seven magnitude bits placed in the float's exponent and mantissa fields read as 2^(e-127) (1+m/8) for a
 * normal code and as the denormal (m/8) 2^-126 for e = 0; times 2^120 both become the code's value (bias 7, subnormal m 2^-9). The sign moves to bit 31. */
__device__ __forceinline__ float gemv_e4m3_fast(uint32_t b) {
    return __uint_as_float(((b << 20) & 0x07F00000u) | ((b << 24) & 0x80000000u)) * 0x1p120f;
}

__constant__ float kGemvW4nl[16] = { 1.125f, 3.25f, 5.5f, 8.0f, 11.0f, 14.0f, 18.0f, 22.0f,
                                     -1.125f, -3.25f, -5.5f, -8.0f, -11.0f, -14.0f, -18.0f, -22.0f };

constexpr uint32_t kGemvTile = 64;        /* outputs a block */
constexpr uint32_t kGemvMaxK = 2560;

template <bool TOKEN, bool FP8>
__global__ __launch_bounds__(256) void lane_moe_gemv(
    const uint8_t* aq, const float* as, const int32_t* sorted, const int32_t* off, uint32_t top_k,
    void* const* code0, void* const* scale0, void* const* code1, void* const* scale1,
    uint32_t n0, uint32_t n1, uint32_t k0, uint32_t k1, uint32_t a_stride, uint32_t out_stride, uint16_t* out,
    uint32_t part_pitch, LaneShape shape) {
    using namespace r3::moe_shm;
    const uint32_t r = blockIdx.x;
    if (r >= (uint32_t)off[shape.experts]) return;
    __shared__ float a[kGemvMaxK];
    __shared__ uint32_t sh_e;
    if (threadIdx.x == 0) {
        uint32_t lo = 0, hi = shape.experts;          /* last e with off[e] <= r */
        while (hi - lo > 1) { const uint32_t mid = (lo + hi) / 2; if ((uint32_t)off[mid] <= r) lo = mid; else hi = mid; }
        sh_e = lo;
    }
    __syncthreads();
    const uint32_t e = sh_e, cls = e & 1u, ix = e >> 1;
    const uint32_t N = cls ? n1 : n0, K = cls ? k1 : k0;
    const uint32_t n_base = blockIdx.y * kGemvTile;
    if (n_base >= N || K > kGemvMaxK) return;
    const uint32_t arow = TOKEN ? (uint32_t)sorted[r] / top_k : r;
    const uint8_t* ar = aq + (uint64_t)arow * a_stride;
    const float* asr = as + (uint64_t)arow * (a_stride / 128u);
    for (uint32_t k = threadIdx.x; k < K; k += blockDim.x) a[k] = gemv_e4m3(ar[k]) * asr[k / 128u];
    __syncthreads();

    const uint8_t* codes = static_cast<const uint8_t*>(cls ? code1[ix] : code0[ix]);
    const uint8_t* scales = static_cast<const uint8_t*>(cls ? scale1[ix] : scale0[ix]);
    if (!codes || !scales) return;
    const uint32_t lane = threadIdx.x % warpSize, wave = threadIdx.x / warpSize, waves = blockDim.x / warpSize;
    if constexpr (FP8) {
        /* FP8 block-scaled experts (RAD13/14): row-major E4M3 codes as the checkpoint stores them, one bf16 scale per 128x128 tile
         * ([ceil(N/128)][ceil(K/128)]). Four codes a dword; a dword never straddles a 128-column tile (4 divides 128), so one
         * scale covers it. The activation row in LDS is already dequantised (E4M3 times its per-128 scale), so the dot product is
         * a plain f32 accumulate and the only scale left is the weight tile's. f32 accumulate, one bf16 narrow. */
        const uint32_t kdw = K / 4u, ktiles = (K + 127u) / 128u;
        for (uint32_t j = wave; j < kGemvTile; j += waves) {
            const uint32_t n = n_base + j;
            if (n >= N) break;
            const uint32_t* crow = reinterpret_cast<const uint32_t*>(codes + (uint64_t)n * K);
            const uint16_t* srow = reinterpret_cast<const uint16_t*>(scales) + (uint64_t)(n / 128u) * ktiles;
            float acc = 0.0f;
            for (uint32_t d = lane; d < kdw; d += warpSize) {
                const uint32_t w = crow[d];
                const float* x = a + 4u * d;
                const float part = x[0] * gemv_e4m3(w & 0xFFu) + x[1] * gemv_e4m3((w >> 8) & 0xFFu) +
                                   x[2] * gemv_e4m3((w >> 16) & 0xFFu) + x[3] * gemv_e4m3(w >> 24);
                acc += part * __uint_as_float(((uint32_t)srow[(4u * d) / 128u]) << 16);
            }
            for (uint32_t o = warpSize / 2; o; o >>= 1) acc += __shfl_down(acc, o);
            if (lane == 0) {
                uint32_t u = __float_as_uint(acc);
                u += 0x7fffu + ((u >> 16) & 1u);
                const uint32_t col = TOKEN && part_pitch && n >= N / 2 ? n - N / 2 + part_pitch : n;
                out[(uint64_t)r * out_stride + col] = (uint16_t)(u >> 16);
            }
        }
        return;
    }
    const uint32_t dwords = K / 8u;
    for (uint32_t j = wave; j < kGemvTile; j += waves) {
        const uint32_t n = n_base + j;
        if (n >= N) break;
        const uint32_t* crow = reinterpret_cast<const uint32_t*>(codes + (uint64_t)n * (K / 2u));
        const uint8_t* srow = scales + (uint64_t)n * (K / 64u);
        float acc = 0.0f;
        for (uint32_t d = lane; d < dwords; d += warpSize) {
            const uint32_t w = crow[d];
            const float* x = a + 8u * d;
            float part = 0.0f;
#pragma unroll
            for (uint32_t b = 0; b < 4; ++b) {
                const uint32_t byte = (w >> (8u * b)) & 0xFFu;
                part += x[b] * kGemvW4nl[byte & 15u] + x[b + 4] * kGemvW4nl[byte >> 4];
            }
            acc += part * gemv_e4m3(srow[(8u * d) / 64u]);
        }
        for (uint32_t o = warpSize / 2; o; o >>= 1) acc += __shfl_down(acc, o);
        if (lane == 0) {
            uint32_t u = __float_as_uint(acc * 0x1p-13f);
            u += 0x7fffu + ((u >> 16) & 1u);
            /* Two parts (gate, up): the grouped GEMM puts part 1 at a fixed column, the lane's widest
             * half (part_pitch), whatever this class's width -- so a narrower class's up half starts there. */
            const uint32_t col = TOKEN && part_pitch && n >= N / 2 ? n - N / 2 + part_pitch : n;
            out[(uint64_t)r * out_stride + col] = (uint16_t)(u >> 16);
        }
    }
}

/* WIDE FP8 DECODE GEMV (RAD14). Same block structure and arithmetic as lane_moe_gemv<TOKEN, true> (one block a (routed row, 64
 * outputs), the row's activation dequantised to f32 in LDS once, one wave an output, f32 accumulate, one bf16 narrow), but each lane
 * issues sixteen-byte weight loads, four independent ones before it uses any, so a wave keeps 2 KiB of weights in flight instead of
 * 128 bytes. A sixteen-code load never straddles a 128-column scale tile (16 divides 128), so one scale covers it. The activation is
 * stored in four planes of float4 (plane q holds codes 4q..4q+3 of every sixteen) so a lane's float4 read is one sixteen-byte LDS
 * access and neighbouring lanes read neighbouring words. The lanes sum the K axis in a different order than lane_moe_gemv, so the
 * result is within f32 rounding of it, not bit-identical. K must be a multiple of 16. */
template <bool TOKEN>
__global__ __launch_bounds__(256) void lane_moe_gemv_wide(
    const uint8_t* aq, const float* as, const int32_t* sorted, const int32_t* off, uint32_t top_k,
    void* const* code0, void* const* scale0, void* const* code1, void* const* scale1,
    uint32_t n0, uint32_t n1, uint32_t k0, uint32_t k1, uint32_t a_stride, uint32_t out_stride, uint16_t* out,
    uint32_t part_pitch, LaneShape shape) {
    using namespace r3::moe_shm;
    const uint32_t r = blockIdx.x;
    if (r >= (uint32_t)off[shape.experts]) return;
    __shared__ float a[kGemvMaxK];
    __shared__ uint32_t sh_e;
    if (threadIdx.x == 0) {
        uint32_t lo = 0, hi = shape.experts;          /* last e with off[e] <= r */
        while (hi - lo > 1) { const uint32_t mid = (lo + hi) / 2; if ((uint32_t)off[mid] <= r) lo = mid; else hi = mid; }
        sh_e = lo;
    }
    __syncthreads();
    const uint32_t e = sh_e, cls = e & 1u, ix = e >> 1;
    const uint32_t N = cls ? n1 : n0, K = cls ? k1 : k0;
    const uint32_t n_base = blockIdx.y * kGemvTile;
    if (n_base >= N || K > kGemvMaxK || (K & 15u)) return;
    const uint32_t arow = TOKEN ? (uint32_t)sorted[r] / top_k : r;
    const uint8_t* ar = aq + (uint64_t)arow * a_stride;
    const float* asr = as + (uint64_t)arow * (a_stride / 128u);
    const uint32_t k16 = K / 16u;
    for (uint32_t k = threadIdx.x; k < K; k += blockDim.x) {
        const uint32_t d = k >> 4, q = (k >> 2) & 3u, c = k & 3u;
        a[(q * k16 + d) * 4u + c] = gemv_e4m3(ar[k]) * asr[k / 128u];
    }
    __syncthreads();

    const uint8_t* codes = static_cast<const uint8_t*>(cls ? code1[ix] : code0[ix]);
    const uint8_t* scales = static_cast<const uint8_t*>(cls ? scale1[ix] : scale0[ix]);
    if (!codes || !scales) return;
    const uint32_t lane = threadIdx.x % warpSize, wave = threadIdx.x / warpSize, waves = blockDim.x / warpSize;
    const uint32_t ktiles = (K + 127u) / 128u;
    constexpr uint32_t U = 4;                         /* independent sixteen-byte loads in flight a lane, for each output */
    constexpr uint32_t O = 2;                         /* outputs a wave handles together, so 2*U loads are in flight */
    const float4* a4 = reinterpret_cast<const float4*>(a);
    for (uint32_t j0 = wave * O; j0 < kGemvTile; j0 += waves * O) {
        const uint4* crow[O];
        const uint16_t* srow[O];
        float acc[O];
#pragma unroll
        for (uint32_t o = 0; o < O; ++o) {
            const uint32_t n = n_base + j0 + o < N ? n_base + j0 + o : N - 1;   /* a clamped row is computed and never stored */
            crow[o] = reinterpret_cast<const uint4*>(codes + (uint64_t)n * K);
            srow[o] = reinterpret_cast<const uint16_t*>(scales) + (uint64_t)(n / 128u) * ktiles;
            acc[o] = 0.0f;
        }
        for (uint32_t d0 = lane; d0 < k16; d0 += U * warpSize) {
            uint4 v[O][U];
#pragma unroll
            for (uint32_t o = 0; o < O; ++o)
#pragma unroll
                for (uint32_t u = 0; u < U; ++u) {
                    const uint32_t d = d0 + u * warpSize;
                    v[o][u] = d < k16 ? crow[o][d] : make_uint4(0u, 0u, 0u, 0u);
                }
#pragma unroll
            for (uint32_t u = 0; u < U; ++u) {
                const uint32_t d = d0 + u * warpSize;
                if (d >= k16) break;
                const float4 x0 = a4[0u * k16 + d], x1 = a4[1u * k16 + d], x2 = a4[2u * k16 + d], x3 = a4[3u * k16 + d];
#pragma unroll
                for (uint32_t o = 0; o < O; ++o) {
                    const uint32_t w0 = v[o][u].x, w1 = v[o][u].y, w2 = v[o][u].z, w3 = v[o][u].w;
                    float part = x0.x * gemv_e4m3_fast(w0) + x0.y * gemv_e4m3_fast(w0 >> 8) +
                                 x0.z * gemv_e4m3_fast(w0 >> 16) + x0.w * gemv_e4m3_fast(w0 >> 24);
                    part += x1.x * gemv_e4m3_fast(w1) + x1.y * gemv_e4m3_fast(w1 >> 8) +
                            x1.z * gemv_e4m3_fast(w1 >> 16) + x1.w * gemv_e4m3_fast(w1 >> 24);
                    part += x2.x * gemv_e4m3_fast(w2) + x2.y * gemv_e4m3_fast(w2 >> 8) +
                            x2.z * gemv_e4m3_fast(w2 >> 16) + x2.w * gemv_e4m3_fast(w2 >> 24);
                    part += x3.x * gemv_e4m3_fast(w3) + x3.y * gemv_e4m3_fast(w3 >> 8) +
                            x3.z * gemv_e4m3_fast(w3 >> 16) + x3.w * gemv_e4m3_fast(w3 >> 24);
                    acc[o] += part * __uint_as_float(((uint32_t)srow[o][(16u * d) / 128u]) << 16);
                }
            }
        }
#pragma unroll
        for (uint32_t o = 0; o < O; ++o) {
            float x = acc[o];
            for (uint32_t m = warpSize / 2; m; m >>= 1) x += __shfl_down(x, m);
            const uint32_t n = n_base + j0 + o;
            if (lane == 0 && n < N) {
                uint32_t u = __float_as_uint(x);
                u += 0x7fffu + ((u >> 16) & 1u);
                const uint32_t col = TOKEN && part_pitch && n >= N / 2 ? n - N / 2 + part_pitch : n;
                out[(uint64_t)r * out_stride + col] = (uint16_t)(u >> 16);
            }
        }
    }
}

/* PACKED-FP16 FP8 DECODE GEMV (RAD14). Same blocks as lane_moe_gemv_wide (one block a (routed row, 64 outputs), one wave an output, sixteen-byte
 * weight loads, four in flight a lane), but the dot product runs on v_dot2_f32_f16: two codes a dot instruction, f32 accumulate. An E4M3 code is
 * exactly an f16 value (the seven magnitude bits placed at bits 13..7 read as 2^(e-15)(1+m/8), times 2^8 for the bias change), and the product
 * of two E4M3 values is exact in f16, so nothing is rounded before the f32 accumulation; the activation's per-128 scale and the weight's
 * per-128x128 scale are applied once to each sixteen-element partial sum. The activation is staged as raw E4M3 widened to f16, in two planes of
 * half2 (plane 0 holds elements 0,2 of each four, plane 1 elements 1,3) so a lane's sixteen-byte LDS read matches the weight dword split
 * (w & 0x00FF00FF, (w >> 8) & 0x00FF00FF). K must be a multiple of 16. */
typedef _Float16 h2v __attribute__((ext_vector_type(2)));
__device__ __forceinline__ h2v e4m3x2_to_h2(uint32_t x) {   /* codes in bits 0..7 and 16..23 */
    const uint32_t h = ((x & 0x007F007Fu) << 7) | ((x & 0x00800080u) << 8);
    const h2v v = __builtin_bit_cast(h2v, h);
    return v * h2v{(_Float16)256.0f, (_Float16)256.0f};
}
/* R rows a block: a block whose sorted row is the first of a pair inside its expert (offset from the expert first row a multiple of R) handles that
 * row and the next one of the same expert, decoding each weight once for both; the follower row block returns at once. R = 1 is the plain kernel. */
template <bool TOKEN, uint32_t R>
__global__ __launch_bounds__(256) void lane_moe_gemv_h2(
    const uint8_t* aq, const float* as, const int32_t* sorted, const int32_t* off, uint32_t top_k,
    void* const* code0, void* const* scale0, void* const* code1, void* const* scale1,
    uint32_t n0, uint32_t n1, uint32_t k0, uint32_t k1, uint32_t a_stride, uint32_t out_stride, uint16_t* out,
    uint32_t part_pitch, LaneShape shape) {
    using namespace r3::moe_shm;
    const uint32_t r = blockIdx.x;
    if (r >= (uint32_t)off[shape.experts]) return;
    __shared__ uint32_t a[R][kGemvMaxK / 2];        /* K half values as half2 pairs, in two planes, a row */
    __shared__ float asc[R][kGemvMaxK / 128u];
    __shared__ uint32_t sh_e, sh_first;
    if (threadIdx.x == 0) {
        uint32_t lo = 0, hi = shape.experts;
        while (hi - lo > 1) { const uint32_t mid = (lo + hi) / 2; if ((uint32_t)off[mid] <= r) lo = mid; else hi = mid; }
        sh_e = lo;
        sh_first = (uint32_t)off[lo];
    }
    __syncthreads();
    const uint32_t e = sh_e, cls = e & 1u, ix = e >> 1;
    if (R > 1 && ((r - sh_first) % R)) return;      /* a follower row: its leader computes it */
    const uint32_t nr = (uint32_t)off[e + 1] - r < R ? (uint32_t)off[e + 1] - r : R;
    const uint32_t N = cls ? n1 : n0, K = cls ? k1 : k0;
    const uint32_t n_base = blockIdx.y * kGemvTile;
    if (n_base >= N || K > kGemvMaxK || (K & 15u)) return;
    const uint32_t k16 = K / 16u;
    /* element k = 16 d + 4 q + j sits in plane (j & 1), half2 index (d * 4 + q), lane (j >> 1) */
    for (uint32_t m = 0; m < nr; ++m) {
        const uint32_t arow = TOKEN ? (uint32_t)sorted[r + m] / top_k : r + m;
        const uint8_t* ar = aq + (uint64_t)arow * a_stride;
        const float* asr = as + (uint64_t)arow * (a_stride / 128u);
        uint16_t* ah = reinterpret_cast<uint16_t*>(a[m]);
        for (uint32_t k = threadIdx.x; k < K; k += blockDim.x) {
            const uint32_t d = k >> 4, q = (k >> 2) & 3u, j = k & 3u;
            const uint32_t h = (((uint32_t)ar[k] & 0x7Fu) << 7) | (((uint32_t)ar[k] & 0x80u) << 8);
            const _Float16 v = __builtin_bit_cast(_Float16, (uint16_t)h) * (_Float16)256.0f;
            ah[((((j & 1u) * k16 + d) * 4u + q) << 1) + (j >> 1)] = __builtin_bit_cast(uint16_t, v);
        }
        for (uint32_t g = threadIdx.x; g < K / 128u; g += blockDim.x) asc[m][g] = asr[g];
    }
    __syncthreads();

    const uint8_t* codes = static_cast<const uint8_t*>(cls ? code1[ix] : code0[ix]);
    const uint8_t* scales = static_cast<const uint8_t*>(cls ? scale1[ix] : scale0[ix]);
    if (!codes || !scales) return;
    const uint32_t lane = threadIdx.x % warpSize, wave = threadIdx.x / warpSize, waves = blockDim.x / warpSize;
    const uint32_t ktiles = (K + 127u) / 128u;
    constexpr uint32_t U = 4;
    for (uint32_t j = wave; j < kGemvTile; j += waves) {
        const uint32_t n = n_base + j;
        if (n >= N) break;
        const uint4* crow = reinterpret_cast<const uint4*>(codes + (uint64_t)n * K);
        const uint16_t* srow = reinterpret_cast<const uint16_t*>(scales) + (uint64_t)(n / 128u) * ktiles;
        float acc[R];
#pragma unroll
        for (uint32_t m = 0; m < R; ++m) acc[m] = 0.0f;
        for (uint32_t d0 = lane; d0 < k16; d0 += U * warpSize) {
            uint4 v[U];
#pragma unroll
            for (uint32_t u = 0; u < U; ++u) {
                const uint32_t d = d0 + u * warpSize;
                v[u] = d < k16 ? crow[d] : make_uint4(0u, 0u, 0u, 0u);
            }
#pragma unroll
            for (uint32_t u = 0; u < U; ++u) {
                const uint32_t d = d0 + u * warpSize;
                if (d >= k16) break;
                const uint32_t ws[4] = {v[u].x, v[u].y, v[u].z, v[u].w};
                h2v we[4], wo[4];                      /* the sixteen weights as eight half2, decoded once for every row */
#pragma unroll
                for (uint32_t q = 0; q < 4; ++q) {
                    we[q] = e4m3x2_to_h2(ws[q] & 0x00FF00FFu);
                    wo[q] = e4m3x2_to_h2((ws[q] >> 8) & 0x00FF00FFu);
                }
                const float sw = __uint_as_float(((uint32_t)srow[(16u * d) / 128u]) << 16);
#pragma unroll
                for (uint32_t m = 0; m < R; ++m) {
                    if (m >= nr) break;
                    const uint4 xe = reinterpret_cast<const uint4*>(a[m])[0u * k16 + d], xo = reinterpret_cast<const uint4*>(a[m])[1u * k16 + d];
                    const uint32_t xes[4] = {xe.x, xe.y, xe.z, xe.w}, xos[4] = {xo.x, xo.y, xo.z, xo.w};
                    float part = 0.0f;
#pragma unroll
                    for (uint32_t q = 0; q < 4; ++q) {
                        part = __builtin_amdgcn_fdot2(we[q], __builtin_bit_cast(h2v, xes[q]), part, false);
                        part = __builtin_amdgcn_fdot2(wo[q], __builtin_bit_cast(h2v, xos[q]), part, false);
                    }
                    acc[m] += part * (sw * asc[m][(16u * d) / 128u]);
                }
            }
        }
#pragma unroll
        for (uint32_t m = 0; m < R; ++m) {
            if (m >= nr) break;
            float x = acc[m];
            for (uint32_t o = warpSize / 2; o; o >>= 1) x += __shfl_down(x, o);
            if (lane == 0) {
                uint32_t u = __float_as_uint(x);
                u += 0x7fffu + ((u >> 16) & 1u);
                const uint32_t col = TOKEN && part_pitch && n >= N / 2 ? n - N / 2 + part_pitch : n;
                out[(uint64_t)(r + m) * out_stride + col] = (uint16_t)(u >> 16);
            }
        }
    }
}

/* PACKED-FP16 W4 DECODE GEMV (RAD14, Flash-Next). lane_moe_gemv<TOKEN, false> rebuilt the way the FP8 kernels were: one block a (routed
 * row, 64 outputs), one wave an output, sixteen-byte weight loads (32 codes) four in flight a lane, and the dot product on v_dot2_f32_f16.
 * A nonlinear level (1.125 .. 22) and an E4M3 activation are both exact in f16 and so is their product, so nothing rounds before the f32
 * accumulation; the per-64 E4M3 weight scale (times 2^-13) and the activation's per-128 scale are applied to each 32-element partial sum.
 * The level lookup is two byte permutes over the f16 table held in registers, not a per-lane load from a constant table.
 * The weight layout is the relayout's (moew4_permute): in a dword, byte b holds element b in its low nibble and element b+4 in its high
 * nibble, so a dword is elements 0..7 and the low nibbles are 0..3. The activation row is staged as f16 in element order, so a dword's
 * eight activations are one sixteen-byte LDS read. K must be a multiple of 64. */
__device__ __forceinline__ void w4nl_to_h2x2(uint32_t nib, h2v* p01, h2v* p23) {   /* four codes, one a byte (bits 0..3) */
    /* f16 of the levels 1.125 3.25 5.5 8 11 14 18 22: low bytes 80 80 80 00 80 00 80 80, high bytes 3C 42 45 48 49 4B 4C 4D */
    const uint32_t idx = nib & 0x07070707u;
    const uint32_t lo = __builtin_amdgcn_perm(0x80800080u, 0x00808080u, idx);   /* selector s < 4 reads src1, 4..7 src0 */
    const uint32_t hi = __builtin_amdgcn_perm(0x4D4C4B49u, 0x4845423Cu, idx);
    const uint32_t sg = __builtin_amdgcn_perm(0u, nib & 0x08080808u, 0x010C000Cu) << 4;   /* sign bits 3 -> 15 and 31 */
    const uint32_t sg2 = __builtin_amdgcn_perm(0u, nib & 0x08080808u, 0x030C020Cu) << 4;
    *p01 = __builtin_bit_cast(h2v, __builtin_amdgcn_perm(hi, lo, 0x05010400u) | sg);
    *p23 = __builtin_bit_cast(h2v, __builtin_amdgcn_perm(hi, lo, 0x07030602u) | sg2);
}
template <bool TOKEN>
__global__ __launch_bounds__(256) void lane_moe_gemv_w4h2(
    const uint8_t* aq, const float* as, const int32_t* sorted, const int32_t* off, uint32_t top_k,
    void* const* code0, void* const* scale0, void* const* code1, void* const* scale1,
    uint32_t n0, uint32_t n1, uint32_t k0, uint32_t k1, uint32_t a_stride, uint32_t out_stride, uint16_t* out,
    uint32_t part_pitch, LaneShape shape) {
    using namespace r3::moe_shm;
    const uint32_t r = blockIdx.x;
    if (r >= (uint32_t)off[shape.experts]) return;
    __shared__ uint16_t a[kGemvMaxK];
    __shared__ float asc[kGemvMaxK / 128u];
    __shared__ uint32_t sh_e;
    if (threadIdx.x == 0) {
        uint32_t lo = 0, hi = shape.experts;
        while (hi - lo > 1) { const uint32_t mid = (lo + hi) / 2; if ((uint32_t)off[mid] <= r) lo = mid; else hi = mid; }
        sh_e = lo;
    }
    __syncthreads();
    const uint32_t e = sh_e, cls = e & 1u, ix = e >> 1;
    const uint32_t N = cls ? n1 : n0, K = cls ? k1 : k0;
    const uint32_t n_base = blockIdx.y * kGemvTile;
    if (n_base >= N || K > kGemvMaxK || (K & 63u)) return;
    const uint32_t arow = TOKEN ? (uint32_t)sorted[r] / top_k : r;
    const uint8_t* ar = aq + (uint64_t)arow * a_stride;
    const float* asr = as + (uint64_t)arow * (a_stride / 128u);
    for (uint32_t k = threadIdx.x; k < K; k += blockDim.x) {
        const uint32_t h = (((uint32_t)ar[k] & 0x7Fu) << 7) | (((uint32_t)ar[k] & 0x80u) << 8);
        a[k] = __builtin_bit_cast(uint16_t, (_Float16)(__builtin_bit_cast(_Float16, (uint16_t)h) * (_Float16)256.0f));
    }
    for (uint32_t g = threadIdx.x; g < K / 128u; g += blockDim.x) asc[g] = asr[g];
    __syncthreads();

    const uint8_t* codes = static_cast<const uint8_t*>(cls ? code1[ix] : code0[ix]);
    const uint8_t* scales = static_cast<const uint8_t*>(cls ? scale1[ix] : scale0[ix]);
    if (!codes || !scales) return;
    const uint32_t lane = threadIdx.x % warpSize, wave = threadIdx.x / warpSize, waves = blockDim.x / warpSize;
    const uint32_t k32 = K / 32u;                 /* sixteen-byte groups of 32 codes in a row */
    constexpr uint32_t U = 4;
    const uint4* a4 = reinterpret_cast<const uint4*>(a);   /* eight f16 activations a uint4 */
    for (uint32_t j = wave; j < kGemvTile; j += waves) {
        const uint32_t n = n_base + j;
        if (n >= N) break;
        const uint4* crow = reinterpret_cast<const uint4*>(codes + (uint64_t)n * (K / 2u));
        const uint8_t* srow = scales + (uint64_t)n * (K / 64u);
        float acc = 0.0f;
        for (uint32_t d0 = lane; d0 < k32; d0 += U * warpSize) {
            uint4 v[U];
#pragma unroll
            for (uint32_t u = 0; u < U; ++u) {
                const uint32_t d = d0 + u * warpSize;
                v[u] = d < k32 ? crow[d] : make_uint4(0u, 0u, 0u, 0u);
            }
#pragma unroll
            for (uint32_t u = 0; u < U; ++u) {
                const uint32_t d = d0 + u * warpSize;
                if (d >= k32) break;
                const uint32_t ws[4] = {v[u].x, v[u].y, v[u].z, v[u].w};
                float part = 0.0f;
#pragma unroll
                for (uint32_t q = 0; q < 4; ++q) {        /* dword q: elements 32d + 8q .. +7 */
                    const uint4 x = a4[4u * d + q];       /* activations e0..e7 as four half2 */
                    h2v w01, w23, w45, w67;
                    w4nl_to_h2x2(ws[q] & 0x0F0F0F0Fu, &w01, &w23);
                    w4nl_to_h2x2((ws[q] >> 4) & 0x0F0F0F0Fu, &w45, &w67);
                    part = __builtin_amdgcn_fdot2(w01, __builtin_bit_cast(h2v, x.x), part, false);
                    part = __builtin_amdgcn_fdot2(w23, __builtin_bit_cast(h2v, x.y), part, false);
                    part = __builtin_amdgcn_fdot2(w45, __builtin_bit_cast(h2v, x.z), part, false);
                    part = __builtin_amdgcn_fdot2(w67, __builtin_bit_cast(h2v, x.w), part, false);
                }
                acc += part * (gemv_e4m3_fast(srow[(32u * d) / 64u]) * asc[(32u * d) / 128u]);
            }
        }
        for (uint32_t o = warpSize / 2; o; o >>= 1) acc += __shfl_down(acc, o);
        if (lane == 0) {
            uint32_t u = __float_as_uint(acc * 0x1p-13f);
            u += 0x7fffu + ((u >> 16) & 1u);
            const uint32_t col = TOKEN && part_pitch && n >= N / 2 ? n - N / 2 + part_pitch : n;
            out[(uint64_t)r * out_stride + col] = (uint16_t)(u >> 16);
        }
    }
}

/* MULTI-ROW WIDE FP8 DECODE GEMV (RAD14). lane_moe_gemv_wide with the expert's rows batched: one block an (expert, 64 outputs) handles the
 * expert's routed rows four at a time, so each sixteen-byte weight load is decoded once and applied to every row of the pass instead of
 * once a route (a 35B decode request sends about five routes to each expert it touches). The four activation rows are staged in LDS in the
 * same float4 planes. Sorted row r keeps its own output slot and its token's activation row, so the result for a row equals the single-row
 * kernel's up to the order of the K sum. */
constexpr uint32_t kMrRows = 4;
constexpr uint32_t kMrTile = 64;         /* outputs a block (16 and 32 were slower: each block restages its activation rows) */
template <bool TOKEN>
__global__ __launch_bounds__(256) void lane_moe_gemv_mr(
    const uint8_t* aq, const float* as, const int32_t* sorted, const int32_t* off, uint32_t top_k,
    void* const* code0, void* const* scale0, void* const* code1, void* const* scale1,
    uint32_t n0, uint32_t n1, uint32_t k0, uint32_t k1, uint32_t a_stride, uint32_t out_stride, uint16_t* out,
    uint32_t part_pitch) {
    using namespace r3::moe_shm;
    const uint32_t e = blockIdx.x;
    const uint32_t rs = (uint32_t)off[e], re = (uint32_t)off[e + 1];
    if (rs >= re) return;
    const uint32_t cls = e & 1u, ix = e >> 1;
    const uint32_t N = cls ? n1 : n0, K = cls ? k1 : k0;
    const uint32_t n_base = blockIdx.y * kMrTile;
    if (n_base >= N || K > kGemvMaxK || (K & 15u)) return;
    const uint8_t* codes = static_cast<const uint8_t*>(cls ? code1[ix] : code0[ix]);
    const uint8_t* scales = static_cast<const uint8_t*>(cls ? scale1[ix] : scale0[ix]);
    if (!codes || !scales) return;
    __shared__ float a[kMrRows][kGemvMaxK];
    const uint32_t k16 = K / 16u, ktiles = (K + 127u) / 128u;
    const uint32_t lane = threadIdx.x % warpSize, wave = threadIdx.x / warpSize, waves = blockDim.x / warpSize;
    constexpr uint32_t U = 4;
    for (uint32_t r0 = rs; r0 < re; r0 += kMrRows) {
        const uint32_t nr = re - r0 < kMrRows ? re - r0 : kMrRows;
        __syncthreads();
        for (uint32_t i = threadIdx.x; i < nr * K; i += blockDim.x) {
            const uint32_t rr = i / K, k = i - rr * K;
            const uint32_t arow = TOKEN ? (uint32_t)sorted[r0 + rr] / top_k : r0 + rr;
            const uint32_t d = k >> 4, q = (k >> 2) & 3u, c = k & 3u;
            a[rr][(q * k16 + d) * 4u + c] = gemv_e4m3(aq[(uint64_t)arow * a_stride + k]) * as[(uint64_t)arow * (a_stride / 128u) + k / 128u];
        }
        __syncthreads();
        for (uint32_t j = wave; j < kMrTile; j += waves) {
            const uint32_t n = n_base + j;
            if (n >= N) break;
            const uint4* crow = reinterpret_cast<const uint4*>(codes + (uint64_t)n * K);
            const uint16_t* srow = reinterpret_cast<const uint16_t*>(scales) + (uint64_t)(n / 128u) * ktiles;
            float acc[kMrRows];
#pragma unroll
            for (uint32_t m = 0; m < kMrRows; ++m) acc[m] = 0.0f;
            for (uint32_t d0 = lane; d0 < k16; d0 += U * warpSize) {
                uint4 v[U];
#pragma unroll
                for (uint32_t u = 0; u < U; ++u) {
                    const uint32_t d = d0 + u * warpSize;
                    v[u] = d < k16 ? crow[d] : make_uint4(0u, 0u, 0u, 0u);
                }
#pragma unroll
                for (uint32_t u = 0; u < U; ++u) {
                    const uint32_t d = d0 + u * warpSize;
                    if (d >= k16) break;
                    float wf[16];
                    const uint32_t ws[4] = {v[u].x, v[u].y, v[u].z, v[u].w};
#pragma unroll
                    for (uint32_t q = 0; q < 4; ++q) {
                        wf[4 * q + 0] = gemv_e4m3_fast(ws[q]);
                        wf[4 * q + 1] = gemv_e4m3_fast(ws[q] >> 8);
                        wf[4 * q + 2] = gemv_e4m3_fast(ws[q] >> 16);
                        wf[4 * q + 3] = gemv_e4m3_fast(ws[q] >> 24);
                    }
                    const float s = __uint_as_float(((uint32_t)srow[(16u * d) / 128u]) << 16);
#pragma unroll
                    for (uint32_t m = 0; m < kMrRows; ++m) {
                        if (m >= nr) break;
                        const float4* a4 = reinterpret_cast<const float4*>(a[m]);
                        float part = 0.0f;
#pragma unroll
                        for (uint32_t q = 0; q < 4; ++q) {
                            const float4 x = a4[q * k16 + d];
                            part += x.x * wf[4 * q] + x.y * wf[4 * q + 1] + x.z * wf[4 * q + 2] + x.w * wf[4 * q + 3];
                        }
                        acc[m] += part * s;
                    }
                }
            }
#pragma unroll
            for (uint32_t m = 0; m < kMrRows; ++m) {
                if (m >= nr) break;
                float x = acc[m];
                for (uint32_t o = warpSize / 2; o; o >>= 1) x += __shfl_down(x, o);
                if (lane == 0) {
                    uint32_t u = __float_as_uint(x);
                    u += 0x7fffu + ((u >> 16) & 1u);
                    const uint32_t col = TOKEN && part_pitch && n >= N / 2 ? n - N / 2 + part_pitch : n;
                    out[(uint64_t)(r0 + m) * out_stride + col] = (uint16_t)(u >> 16);
                }
            }
        }
    }
}

/* PREFILL MoE GEMM FOR FEW ROWS A EXPERT (RAD06), the helper's large lane. With the hot experts on the engine, a
 * helper's experts are cold: a handful of routed rows each per prefill step, where the grouped GEMM's 16-row
 * WMMA tiles are mostly padding. Here one block an (expert, 32 outputs): the expert's packed weight rows are
 * streamed once per 256-wide K chunk and each decoded weight is applied to up to 16 of the expert's rows,
 * whose activations sit dequantised in LDS. Same weight form, table, scales, sorted/off and part-pitch rule as
 * lane_moe_gemv; f32 accumulate, one bf16 narrow. lane_moe_rows_fp8 is the FP8 block-scaled form (f32 operands); lane_moe_rows_w4 the
 * w4nl form (fp16 operands, v_dot2_f32_f16), each with its own tile shape. */
constexpr uint32_t kRowsTileN = 32;       /* outputs a block: 8 waves x 4 */
constexpr uint32_t kRowsOut = 4;          /* outputs a wave */
constexpr uint32_t kRowsRB = 12;          /* rows a pass: 9 rows an expert fits one pass (8 forced a second K traversal for the ninth) */
constexpr uint32_t kRowsKC = 256;         /* K a chunk */

template <bool TOKEN>
__global__ __launch_bounds__(256) void lane_moe_rows_fp8(
    const uint8_t* aq, const float* as, const int32_t* sorted, const int32_t* off, uint32_t top_k,
    void* const* code0, void* const* scale0, void* const* code1, void* const* scale1,
    uint32_t n0, uint32_t n1, uint32_t k0, uint32_t k1, uint32_t a_stride, uint32_t out_stride, uint16_t* out,
    uint32_t part_pitch) {
    using namespace r3::moe_shm;
    const uint32_t e = blockIdx.x;
    const uint32_t rs = (uint32_t)off[e], re = (uint32_t)off[e + 1];
    if (rs >= re) return;
    const uint32_t cls = e & 1u, ix = e >> 1;
    const uint32_t N = cls ? n1 : n0, K = cls ? k1 : k0;
    const uint32_t n_base = blockIdx.y * kRowsTileN;
    if (n_base >= N || K % 4u) return;
    const uint8_t* codes = static_cast<const uint8_t*>(cls ? code1[ix] : code0[ix]);
    const uint8_t* scales = static_cast<const uint8_t*>(cls ? scale1[ix] : scale0[ix]);
    if (!codes || !scales) return;

    __shared__ float a[kRowsRB][kRowsKC];
    __shared__ uint32_t arow[kRowsRB];
    const uint32_t lane = threadIdx.x % warpSize, wave = threadIdx.x / warpSize;
    const uint32_t n_first = n_base + wave * kRowsOut;

    for (uint32_t r0 = rs; r0 < re; r0 += kRowsRB) {
        const uint32_t nr = re - r0 < kRowsRB ? re - r0 : kRowsRB;
        if (threadIdx.x < nr) arow[threadIdx.x] = TOKEN ? (uint32_t)sorted[r0 + threadIdx.x] / top_k : r0 + threadIdx.x;
        float acc[kRowsOut][kRowsRB];
#pragma unroll
        for (uint32_t j = 0; j < kRowsOut; ++j)
#pragma unroll
            for (uint32_t r = 0; r < kRowsRB; ++r) acc[j][r] = 0.0f;

        for (uint32_t kc = 0; kc < K; kc += kRowsKC) {
            const uint32_t kl = K - kc < kRowsKC ? K - kc : kRowsKC;   /* the last chunk may be short */
            __syncthreads();
            {
                /* the chunk's activations, 16 codes a thread: one 16-byte load of codes and one of their scale (16 codes never straddle a
                 * 128-wide scale group), against a byte load and a scale load per code before */
                static_assert(kRowsKC % 16u == 0 && kRowsRB * (kRowsKC / 16u) <= 256u, "one 16-code piece a thread");
                const uint32_t pieces = kRowsKC / 16u;
                const uint32_t r = threadIdx.x / pieces, p16 = threadIdx.x % pieces;
                if (r < nr && 16u * p16 < kl) {
                    const uint32_t ar = arow[r], k = kc + 16u * p16;
                    const uint4 q = *reinterpret_cast<const uint4*>(aq + (uint64_t)ar * a_stride + k);
                    const float sc = as[(uint64_t)ar * (a_stride / 128u) + k / 128u];
                    const uint32_t qw[4] = {q.x, q.y, q.z, q.w};
                    float* dst = &a[r][16u * p16];
#pragma unroll
                    for (uint32_t w = 0; w < 4; ++w) {
                        float4 v;
                        v.x = gemv_e4m3(qw[w] & 0xFFu) * sc; v.y = gemv_e4m3((qw[w] >> 8) & 0xFFu) * sc;
                        v.z = gemv_e4m3((qw[w] >> 16) & 0xFFu) * sc; v.w = gemv_e4m3(qw[w] >> 24) * sc;
                        *reinterpret_cast<float4*>(dst + 4u * w) = v;
                    }
                }
            }
            __syncthreads();
            {
                /* FP8 block-scaled experts (RAD14): E4M3 codes row-major as the checkpoint stores them, one bf16 scale per 128x128
                 * tile. Each dword is four codes; a dword never straddles a tile (4 divides 128). The code is decoded once and
                 * applied to every row of the expert in this pass, so the weights are read once an expert rather than once a
                 * route. The weight scale is folded into the decoded values, so the rounding differs from lane_moe_gemv. */
                const uint32_t ktiles = (K + 127u) / 128u;
                for (uint32_t d = lane; d < kl / 4u; d += warpSize) {
#pragma unroll
                    for (uint32_t j = 0; j < kRowsOut; ++j) {
                        const uint32_t n = n_first + j;
                        if (n >= N) break;
                        const uint32_t w = reinterpret_cast<const uint32_t*>(codes + (uint64_t)n * K)[kc / 4u + d];
                        const float s = __uint_as_float(
                            ((uint32_t)reinterpret_cast<const uint16_t*>(scales)[(uint64_t)(n / 128u) * ktiles + (kc + 4u * d) / 128u]) << 16);
                        const float w0 = gemv_e4m3(w & 0xFFu) * s, w1 = gemv_e4m3((w >> 8) & 0xFFu) * s,
                                    w2 = gemv_e4m3((w >> 16) & 0xFFu) * s, w3 = gemv_e4m3(w >> 24) * s;
#pragma unroll
                        for (uint32_t r = 0; r < kRowsRB; ++r) {
                            if (r >= nr) continue;
                            const float* x = &a[r][4u * d];
                            acc[j][r] += x[0] * w0 + x[1] * w1 + x[2] * w2 + x[3] * w3;
                        }
                    }
                }
            }
        }
#pragma unroll
        for (uint32_t j = 0; j < kRowsOut; ++j) {
            const uint32_t n = n_first + j;
#pragma unroll
            for (uint32_t r = 0; r < kRowsRB; ++r) {
                if (r >= nr) continue;
                float v = acc[j][r];
                for (uint32_t o = warpSize / 2; o; o >>= 1) v += __shfl_down(v, o);
                if (lane == 0 && n < N) {
                    uint32_t u = __float_as_uint(v);
                    u += 0x7fffu + ((u >> 16) & 1u);
                    const uint32_t col = TOKEN && part_pitch && n >= N / 2 ? n - N / 2 + part_pitch : n;
                    out[(uint64_t)(r0 + r) * out_stride + col] = (uint16_t)(u >> 16);
                }
            }
        }
    }
}


/* RAD46 output-owned-lane experiment. Host dispatch restricts this kernel to the validated W4
 * geometry; the existing rows kernel remains the fallback. One thread owns one output column,
 * reusing its decoded weights over RB rows. Packed weights are loaded cooperatively along K,
 * then transposed in LDS so output lanes read adjacent columns. The padded column pitch avoids
 * the worst transpose bank conflicts. Activation and weight tiles are double-buffered.
 *
 * Arithmetic is the group-fold experiment's: two independently reset dot8 values form each
 * dot16, four dot16 values accumulate sequentially per 64 K, then two weighted groups fold
 * under one activation scale per 128 K. No cross-lane arithmetic and no final reduction.
 * This preserves group-fold's operation order; it does not claim libr2 equivalence on all inputs. */
constexpr uint32_t kOutputLaneN = 128, kOutputLaneK = 128, kOutputLaneRB = 8;
template <bool TOKEN, uint32_t RB>
__global__ __launch_bounds__(128) void lane_moe_rows_w4_outputlane(
    const uint8_t* aq, const float* as, const int32_t* sorted, const int32_t* off, uint32_t top_k,
    void* const* code0, void* const* scale0, void* const* code1, void* const* scale1,
    uint32_t n0, uint32_t n1, uint32_t k0, uint32_t k1, uint32_t a_stride, uint32_t out_stride,
    uint16_t* out, uint32_t part_pitch, uint32_t lo, uint32_t hi) {
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
    static_assert(RB > 0 && RB <= 16, "output-lane row batch must fit the staging geometry");
    const uint32_t e = blockIdx.x, tid = threadIdx.x;
    const uint32_t rs = (uint32_t)off[e], re = (uint32_t)off[e + 1];
    if (rs >= re || re - rs < lo || re - rs >= hi) return; // TK03: this launch serves experts with lo <= routes < hi
    const uint32_t cls = e & 1u, ix = e >> 1;
    const uint32_t N = cls ? n1 : n0, K = cls ? k1 : k0;
    const uint32_t n_base = blockIdx.y * kOutputLaneN, n = n_base + tid;
    if (n_base >= N) return;
    const uint8_t* codes = static_cast<const uint8_t*>(cls ? code1[ix] : code0[ix]);
    const uint8_t* scales = static_cast<const uint8_t*>(cls ? scale1[ix] : scale0[ix]);
    if (!codes || !scales) return;
    constexpr uint32_t words = kOutputLaneK / 8;
    __shared__ uint32_t weight[2][words][kOutputLaneN + 1];
    __shared__ uint4 activation[2][RB][words];
    __shared__ float activation_scale[2][RB];
    __shared__ uint32_t arow[RB];
    __shared__ float lut[16];
    if (tid < 16) lut[tid] = kGemvW4nl[tid];
    auto pack = [](_Float16 x, _Float16 y) {
        return (uint32_t)__builtin_bit_cast(uint16_t, x) |
               ((uint32_t)__builtin_bit_cast(uint16_t, y) << 16);
    };
    auto decode = [&](uint32_t w) {
        // Packed dword order: low nibbles are K0..3, high nibbles are K4..7.
        return make_uint4(
            pack((_Float16)lut[w & 15u], (_Float16)lut[(w >> 8) & 15u]),
            pack((_Float16)lut[(w >> 16) & 15u], (_Float16)lut[(w >> 24) & 15u]),
            pack((_Float16)lut[(w >> 4) & 15u], (_Float16)lut[(w >> 12) & 15u]),
            pack((_Float16)lut[(w >> 20) & 15u], (_Float16)lut[(w >> 28) & 15u]));
    };
    auto dot8 = [](uint4 x, uint4 w) {
        float s = 0.0f;
        asm("v_dot2_f32_f16 %0, %1, %2, %0" : "+v"(s) : "v"(x.x), "v"(w.x));
        asm("v_dot2_f32_f16 %0, %1, %2, %0" : "+v"(s) : "v"(x.y), "v"(w.y));
        asm("v_dot2_f32_f16 %0, %1, %2, %0" : "+v"(s) : "v"(x.z), "v"(w.z));
        asm("v_dot2_f32_f16 %0, %1, %2, %0" : "+v"(s) : "v"(x.w), "v"(w.w));
        return s;
    };
    for (uint32_t r0 = rs; r0 < re; r0 += RB) {
        const uint32_t nr = re - r0 < RB ? re - r0 : RB;
        if (tid < nr) arow[tid] = TOKEN ? (uint32_t)sorted[r0 + tid] / top_k : r0 + tid;
        float acc[RB];
#pragma unroll
        for (uint32_t r = 0; r < RB; ++r) acc[r] = 0.0f;
        uint32_t prefetched[words];
        uint16_t prefetched_scales;
        auto fetch = [&](uint32_t kc) {
            // A half wave reads sixteen consecutive packed words from one output row.
#pragma unroll
            for (uint32_t p = 0; p < words; ++p) {
                const uint32_t i = tid + p * kOutputLaneN, col = i / words, kw = i % words;
                const uint32_t wn = n_base + col;
                prefetched[p] = wn < N ? reinterpret_cast<const uint32_t*>(
                    codes + (uint64_t)wn * (K / 2u))[kc / 8u + kw] : 0u;
            }
            // Both 64-K scales of this thread's column, already in K order.
            prefetched_scales = n < N ? reinterpret_cast<const uint16_t*>(
                scales + (uint64_t)n * (K / 64u))[kc / kOutputLaneK] : (uint16_t)0;
        };
        auto stage = [&](uint32_t kc, uint32_t buf) {
#pragma unroll
            for (uint32_t p = 0; p < words; ++p) {
                const uint32_t i = tid + p * kOutputLaneN;
                weight[buf][i % words][i / words] = prefetched[p];
            }
            if (tid < nr)
                activation_scale[buf][tid] = as[(uint64_t)arow[tid] * (a_stride / 128u) + kc / 128u];
            for (uint32_t i = tid; i < RB * (kOutputLaneK / 16u); i += kOutputLaneN) {
                const uint32_t r = i / (kOutputLaneK / 16u), p16 = i % (kOutputLaneK / 16u);
                if (r >= nr) continue;
                const uint4 q = *reinterpret_cast<const uint4*>(
                    aq + (uint64_t)arow[r] * a_stride + kc + 16u * p16);
                const uint32_t qw[4] = {q.x, q.y, q.z, q.w};
                uint32_t hw[8];
#pragma unroll
                for (uint32_t j = 0; j < 4; ++j) {
                    hw[2 * j] = pack((_Float16)gemv_e4m3(qw[j] & 255u),
                                     (_Float16)gemv_e4m3((qw[j] >> 8) & 255u));
                    hw[2 * j + 1] = pack((_Float16)gemv_e4m3((qw[j] >> 16) & 255u),
                                         (_Float16)gemv_e4m3(qw[j] >> 24));
                }
                activation[buf][r][2 * p16] = make_uint4(hw[0], hw[1], hw[2], hw[3]);
                activation[buf][r][2 * p16 + 1] = make_uint4(hw[4], hw[5], hw[6], hw[7]);
            }
        };
        fetch(0);
        __syncthreads(); // row map, LUT, and the preceding row batch's shared reads
        stage(0, 0);
        __syncthreads();
        uint32_t cur = 0;
        // Deliberately leave K rolled; each chunk's two groups/four partials are fixed-size.
        for (uint32_t kc = 0; kc < K; kc += kOutputLaneK, cur ^= 1u) {
            const uint16_t ws = prefetched_scales;
            const float ws0 = gemv_e4m3(ws & 255u) * 0x1p-13f;
            const float ws1 = gemv_e4m3(ws >> 8) * 0x1p-13f;
            if (kc + kOutputLaneK < K) fetch(kc + kOutputLaneK);
            float first_weighted[RB];
#pragma unroll
            for (uint32_t group = 0; group < 2; ++group) {
                float raw[RB];
#pragma unroll
                for (uint32_t pair = 0; pair < 4; ++pair) {
                    const uint32_t kw = group * 8u + pair * 2u;
                    const uint4 w0 = decode(weight[cur][kw][tid]);
                    const uint4 w1 = decode(weight[cur][kw + 1u][tid]);
#pragma unroll
                    for (uint32_t r = 0; r < RB; ++r) {
                        if (r >= nr) continue;
                        const float lo = dot8(activation[cur][r][kw], w0);
                        const float hi = dot8(activation[cur][r][kw + 1u], w1);
                        const float partial = lo + hi;
                        if (pair == 0) raw[r] = partial;
                        else raw[r] += partial;
                    }
                }
#pragma unroll
                for (uint32_t r = 0; r < RB; ++r) {
                    if (r >= nr) continue;
                    if (group == 0) first_weighted[r] = raw[r] * ws0;
                    else {
                        const float second_weighted = raw[r] * ws1;
                        const float folded = first_weighted[r] + second_weighted;
                        acc[r] += folded * activation_scale[cur][r];
                    }
                }
            }
            if (kc + kOutputLaneK < K) stage(kc + kOutputLaneK, cur ^ 1u);
            __syncthreads();
        }
        if (n < N) {
            const uint32_t col = TOKEN && part_pitch && n >= N / 2 ? n - N / 2 + part_pitch : n;
#pragma unroll
            for (uint32_t r = 0; r < RB; ++r) {
                if (r >= nr) continue;
                uint32_t u = __float_as_uint(acc[r]);
                u += 0x7fffu + ((u >> 16) & 1u);
                out[(uint64_t)(r0 + r) * out_stride + col] = (uint16_t)(u >> 16);
            }
        }
    }
}

/* RAD46 quad-DPP experiment: four adjacent lanes own one output column. Each
 * computes one dot16 in a K64 group. Only quad lane zero's ordered DPP fold is
 * used; other lanes must remain active as DPP sources, including padded rows.
 * Dot8 resets, dot16 additions, K64 ordering and the K128 scale fold match the
 * output-lane experiment. This is not a claim of universal libr2 equivalence.
 * Host dispatch admits only complete K128 groups and complete N32 tiles. */
constexpr uint32_t kQuadDppN = 32, kQuadDppRB = 16;
// RAD46 v2: halve the 16-row live accumulation/staging set for gate/up.
// Each row has an independent, bit-identical K-fold; the down path keeps the
// established 16-row batch until its separate regression gate is measured.
constexpr uint32_t kQuadV2GuRB = kQuadDppRB / 2;
// RAD49 (TK01/TK02): W4 levels decode with the register byte-permute decoder (w4nl_to_h2x2, the GEMV's) -- the exact f16
// bits of the same levels in the same element order as the former LDS table -- and each quad broadcast is folded into its
// add (v_add_f32_dpp): the same operands in swapped order, exact for an fp32 add. Bit-identical to the RAD46 form on 40
// captured tail requests (runs/rad49-tail-quadopt), 9% less GEMM time on gfx1030.
template <bool TOKEN, uint32_t RB, uint32_t STAGE = 0>
__global__ __launch_bounds__(128) void lane_moe_rows_w4_quaddpp(
    const uint8_t* aq, const float* as, const int32_t* sorted, const int32_t* off, uint32_t top_k,
    void* const* code0, void* const* scale0, void* const* code1, void* const* scale1,
    uint32_t n0, uint32_t n1, uint32_t k0, uint32_t k1, uint32_t a_stride, uint32_t out_stride,
    uint16_t* out, uint32_t part_pitch, uint32_t lo, uint32_t hi) {
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
    static_assert(RB > 0 && RB <= 16, "quad-DPP row batch exceeds staging geometry");
    const uint32_t e = blockIdx.x, tid = threadIdx.x, qlane = tid & 3u;
    const uint32_t rs = (uint32_t)off[e], re = (uint32_t)off[e + 1];
    if (rs >= re || re - rs < lo || re - rs >= hi) return; // TK03: this launch serves experts with lo <= routes < hi
    const uint32_t cls = e & 1u, ix = e >> 1;
    const uint32_t N = cls ? n1 : n0, K = cls ? k1 : k0;
    const uint32_t n_base = blockIdx.y * kQuadDppN, n = n_base + tid / 4u;
    if (n_base >= N) return;
    const uint8_t* codes = static_cast<const uint8_t*>(cls ? code1[ix] : code0[ix]);
    const uint8_t* scales = static_cast<const uint8_t*>(cls ? scale1[ix] : scale0[ix]);
    if (!codes || !scales) return;
    __shared__ uint4 activation[2][RB][16];
    __shared__ float activation_scale[2][RB];
    __shared__ uint32_t arow[RB];
    auto pack = [](_Float16 x, _Float16 y) {
        return (uint32_t)__builtin_bit_cast(uint16_t, x) |
               ((uint32_t)__builtin_bit_cast(uint16_t, y) << 16);
    };
    auto decode = [](uint32_t w) {
        h2v a01, a23, b01, b23;
        w4nl_to_h2x2(w & 0x0F0F0F0Fu, &a01, &a23);
        w4nl_to_h2x2((w >> 4) & 0x0F0F0F0Fu, &b01, &b23);
        return make_uint4(__builtin_bit_cast(uint32_t, a01), __builtin_bit_cast(uint32_t, a23),
                          __builtin_bit_cast(uint32_t, b01), __builtin_bit_cast(uint32_t, b23));
    };
    auto dot8 = [](uint4 x, uint4 w) {
        float s = 0.0f;
        asm("v_dot2_f32_f16 %0, %1, %2, %0" : "+v"(s) : "v"(x.x), "v"(w.x));
        asm("v_dot2_f32_f16 %0, %1, %2, %0" : "+v"(s) : "v"(x.y), "v"(w.y));
        asm("v_dot2_f32_f16 %0, %1, %2, %0" : "+v"(s) : "v"(x.z), "v"(w.z));
        asm("v_dot2_f32_f16 %0, %1, %2, %0" : "+v"(s) : "v"(x.w), "v"(w.w));
        return s;
    };
    auto quad_sum = [](float partial) {
        // One asm block keeps each DPP add next to its source; LLVM cannot hoist broadcasts across next-tile staging. Lane q
        // sums partial[q] + partial[1], + partial[2], + partial[3] in that order (the DPP operand is src0, the running sum
        // src1: fp32 add is commutative, so the order of operations matches the former mov+add form bit for bit). The
        // early-clobber keeps the sum off the source register. All four source lanes stay active.
        float sum;
        asm("v_add_f32_dpp %0, %1, %1 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf\n\t"
            "v_add_f32_dpp %0, %1, %0 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf\n\t"
            "v_add_f32_dpp %0, %1, %0 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf"
            : "=&v"(sum) : "v"(partial));
        return sum;
    };
    for (uint32_t r0 = rs; r0 < re; r0 += RB) {
        const uint32_t nr = re - r0 < RB ? re - r0 : RB;
        if (tid < RB)
            arow[tid] = tid < nr ? (TOKEN ? (uint32_t)sorted[r0 + tid] / top_k : r0 + tid) : 0u;
        float acc[RB];
#pragma unroll
        for (uint32_t r = 0; r < RB; ++r) acc[r] = 0.0f;
        uint2 prefetched[2];
        uint16_t prefetched_scales;
        auto fetch = [&](uint32_t kc) {
#pragma unroll
            for (uint32_t group = 0; group < 2; ++group)
                prefetched[group] = *reinterpret_cast<const uint2*>(
                    codes + (uint64_t)n * (K / 2u) + kc / 2u + group * 32u + qlane * 8u);
            prefetched_scales = reinterpret_cast<const uint16_t*>(
                scales + (uint64_t)n * (K / 64u))[kc / 128u];
        };
        auto stage = [&](uint32_t kc, uint32_t buf) {
            if (tid < RB)
                activation_scale[buf][tid] = tid < nr ?
                    as[(uint64_t)arow[tid] * (a_stride / 128u) + kc / 128u] : 0.0f;
            for (uint32_t i = tid; i < RB * 8u; i += 128u) {
                const uint32_t r = i / 8u, p16 = i % 8u;
                // Zero every invalid row, so the hot arithmetic has no row predicates.
                const uint4 q = r < nr ? *reinterpret_cast<const uint4*>(
                    aq + (uint64_t)arow[r] * a_stride + kc + p16 * 16u) : make_uint4(0, 0, 0, 0);
                const uint32_t qw[4] = {q.x, q.y, q.z, q.w};
                uint32_t hw[8];
#pragma unroll
                for (uint32_t j = 0; j < 4; ++j) {
                    if constexpr (STAGE == 1u) { // TK04 timing experiment: no E4M3 conversion, input-dependent garbage; output invalid
                        hw[2 * j] = qw[j]; hw[2 * j + 1] = qw[j] ^ 0x80008000u; continue;
                    }
                    if constexpr (STAGE == 2u) { // TK04: packed exact E4M3 -> fp16: the 7 magnitude bits placed as an fp16 exponent/mantissa
                        // read as 2^(e-15)(1+m/8) (denormal m/8 2^-14 for e = 0); times 2^8 both are the code's value.
                        const uint32_t t01 = (qw[j] & 0xFFu) | ((qw[j] << 8) & 0x00FF0000u);
                        const uint32_t t23 = ((qw[j] >> 16) & 0xFFu) | ((qw[j] >> 8) & 0x00FF0000u);
                        const h2v k256 = {(_Float16)256.0f, (_Float16)256.0f};
                        const uint32_t b01 = ((t01 & 0x007F007Fu) << 7) | ((t01 & 0x00800080u) << 8);
                        const uint32_t b23 = ((t23 & 0x007F007Fu) << 7) | ((t23 & 0x00800080u) << 8);
                        hw[2 * j] = __builtin_bit_cast(uint32_t, __builtin_bit_cast(h2v, b01) * k256);
                        hw[2 * j + 1] = __builtin_bit_cast(uint32_t, __builtin_bit_cast(h2v, b23) * k256);
                        continue;
                    }
                    hw[2 * j] = pack((_Float16)gemv_e4m3(qw[j] & 255u),
                                    (_Float16)gemv_e4m3((qw[j] >> 8) & 255u));
                    hw[2 * j + 1] = pack((_Float16)gemv_e4m3((qw[j] >> 16) & 255u),
                                        (_Float16)gemv_e4m3(qw[j] >> 24));
                }
                activation[buf][r][2 * p16] = make_uint4(hw[0], hw[1], hw[2], hw[3]);
                activation[buf][r][2 * p16 + 1] = make_uint4(hw[4], hw[5], hw[6], hw[7]);
            }
        };
        fetch(0);
        __syncthreads();
        stage(0, 0);
        __syncthreads();
        uint32_t cur = 0;
        for (uint32_t kc = 0; kc < K; kc += 128u, cur ^= 1u) {
            const uint2 current[2] = {prefetched[0], prefetched[1]};
            const uint16_t ws = prefetched_scales;
            const float ws0 = gemv_e4m3(ws & 255u) * 0x1p-13f;
            const float ws1 = gemv_e4m3(ws >> 8) * 0x1p-13f;
            if (kc + 128u < K) fetch(kc + 128u);
            float first_weighted[RB];
#pragma unroll
            for (uint32_t group = 0; group < 2; ++group) {
                const uint4 w0 = decode(current[group].x), w1 = decode(current[group].y);
                const uint32_t kw = group * 8u + qlane * 2u;
#pragma unroll
                for (uint32_t r = 0; r < RB; ++r) {
                    const float lo = dot8(activation[cur][r][kw], w0);
                    const float hi = dot8(activation[cur][r][kw + 1u], w1);
                    const float raw = quad_sum(lo + hi);
                    if (group == 0) first_weighted[r] = raw * ws0;
                    else {
                        const float second_weighted = raw * ws1;
                        const float folded = first_weighted[r] + second_weighted;
                        acc[r] += folded * activation_scale[cur][r];
                    }
                }
            }
            if (kc + 128u < K) stage(kc + 128u, cur ^ 1u);
            __syncthreads();
        }
        if (qlane == 0) {
            const uint32_t col = TOKEN && part_pitch && n >= N / 2 ? n - N / 2 + part_pitch : n;
#pragma unroll
            for (uint32_t r = 0; r < RB; ++r) {
                if (r >= nr) continue;
                uint32_t u = __float_as_uint(acc[r]);
                u += 0x7fffu + ((u >> 16) & 1u);
                out[(uint64_t)(r0 + r) * out_stride + col] = (uint16_t)(u >> 16);
            }
        }
    }
}


/* RAD46 row-owned quad: each 4-lane group computes ONE output for four independent
 * routed rows at a time. Each lane loads its own two K8 weight words per K64;
 * quad shuffles share these PACKED weights among the row owners. Crucially,
 * weight lookup/decoding occurs once per K16 and is reused across RB/4 rows.
 *
 * Unlike the K-partitioned quad kernel, no DPP reduction is performed for
 * every row and K64 group. Each row owner constructs the EXACT four ordered
 * dot16 partials (independent dot8 resets), applies each K64 weight scale
 * in f32, folds the two K64 groups, applies the K128 activation scale in
 * f32, then narrows BF16 once. Each quartet has exactly one output column.
 * Measured speed and bit identity must be checked with --replay-sweep before
 * promotion; no normal worker request dispatches this candidate. */
template <bool TOKEN, uint32_t RB>
__global__ __launch_bounds__(128) void lane_moe_rows_w4_rowlane(
    const uint8_t* aq, const float* as, const int32_t* sorted, const int32_t* off, uint32_t top_k,
    void* const* code0, void* const* scale0, void* const* code1, void* const* scale1,
    uint32_t n0, uint32_t n1, uint32_t k0, uint32_t k1, uint32_t a_stride, uint32_t out_stride,
    uint16_t* out, uint32_t part_pitch) {
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
    static_assert(RB >= 4 && RB <= kQuadDppRB && RB % 4 == 0, "row-lane four rows per output");
    const uint32_t e = blockIdx.x, tid = threadIdx.x, qlane = tid & 3u;
    const uint32_t rs = (uint32_t)off[e], re = (uint32_t)off[e + 1];
    if (rs >= re) return;
    const uint32_t cls = e & 1u, ix = e >> 1;
    const uint32_t N = cls ? n1 : n0, K = cls ? k1 : k0;
    const uint32_t n_base = blockIdx.y * kQuadDppN, n = n_base + tid / 4u;
    if (n_base >= N) return;
    const uint8_t* codes = static_cast<const uint8_t*>(cls ? code1[ix] : code0[ix]);
    const uint8_t* scales = static_cast<const uint8_t*>(cls ? scale1[ix] : scale0[ix]);
    if (!codes || !scales) return;

    __shared__ uint4 activation[2][RB][16];
    __shared__ float activation_scale[2][RB];
    __shared__ uint32_t arow[RB], lut[16];
    if (tid < 16) lut[tid] = __builtin_bit_cast(uint16_t, (_Float16)kGemvW4nl[tid]);
    auto pack = [](_Float16 x, _Float16 y) {
        return (uint32_t)__builtin_bit_cast(uint16_t, x) |
               ((uint32_t)__builtin_bit_cast(uint16_t, y) << 16);
    };
    auto decode = [&](uint32_t w) {
        return make_uint4(
            lut[w & 15u] | (lut[(w >> 8) & 15u] << 16),
            lut[(w >> 16) & 15u] | (lut[(w >> 24) & 15u] << 16),
            lut[(w >> 4) & 15u] | (lut[(w >> 12) & 15u] << 16),
            lut[(w >> 20) & 15u] | (lut[(w >> 28) & 15u] << 16));
    };
    auto dot8 = [](uint4 x, uint4 w) {
        float s = 0.0f;
        asm("v_dot2_f32_f16 %0, %1, %2, %0" : "+v"(s) : "v"(x.x), "v"(w.x));
        asm("v_dot2_f32_f16 %0, %1, %2, %0" : "+v"(s) : "v"(x.y), "v"(w.y));
        asm("v_dot2_f32_f16 %0, %1, %2, %0" : "+v"(s) : "v"(x.z), "v"(w.z));
        asm("v_dot2_f32_f16 %0, %1, %2, %0" : "+v"(s) : "v"(x.w), "v"(w.w));
        return s;
    };
    for (uint32_t r0 = rs; r0 < re; r0 += RB) {
        const uint32_t nr = re - r0 < RB ? re - r0 : RB;
        if (tid < RB)
            arow[tid] = tid < nr ? (TOKEN ? (uint32_t)sorted[r0 + tid] / top_k : r0 + tid) : 0u;
        float acc[RB / 4];
#pragma unroll
        for (uint32_t rb = 0; rb < RB / 4; ++rb) acc[rb] = 0.0f;
        uint2 prefetched[2];
        uint16_t prefetched_scales;
        auto fetch = [&](uint32_t kc) {
#pragma unroll
            for (uint32_t group = 0; group < 2; ++group)
                prefetched[group] = *reinterpret_cast<const uint2*>(
                    codes + (uint64_t)n * (K / 2u) + kc / 2u + group * 32u + qlane * 8u);
            prefetched_scales = reinterpret_cast<const uint16_t*>(
                scales + (uint64_t)n * (K / 64u))[kc / 128u];
        };
        auto stage = [&](uint32_t kc, uint32_t buf) {
            if (tid < RB)
                activation_scale[buf][tid] = tid < nr ?
                    as[(uint64_t)arow[tid] * (a_stride / 128u) + kc / 128u] : 0.0f;
            for (uint32_t i = tid; i < RB * 8u; i += 128u) {
                const uint32_t r = i / 8u, p16 = i % 8u;
                const uint4 q = r < nr ? *reinterpret_cast<const uint4*>(
                    aq + (uint64_t)arow[r] * a_stride + kc + p16 * 16u) : make_uint4(0, 0, 0, 0);
                const uint32_t qw[4] = {q.x, q.y, q.z, q.w};
                uint32_t hw[8];
#pragma unroll
                for (uint32_t j = 0; j < 4; ++j) {
                    hw[2 * j] = pack((_Float16)gemv_e4m3(qw[j] & 255u),
                                     (_Float16)gemv_e4m3((qw[j] >> 8) & 255u));
                    hw[2 * j + 1] = pack((_Float16)gemv_e4m3((qw[j] >> 16) & 255u),
                                         (_Float16)gemv_e4m3(qw[j] >> 24));
                }
                activation[buf][r][2 * p16] = make_uint4(hw[0], hw[1], hw[2], hw[3]);
                activation[buf][r][2 * p16 + 1] = make_uint4(hw[4], hw[5], hw[6], hw[7]);
            }
        };
        fetch(0);
        __syncthreads();
        stage(0, 0);
        __syncthreads();
        uint32_t cur = 0;
        for (uint32_t kc = 0; kc < K; kc += 128u, cur ^= 1u) {
            const uint2 current[2] = {prefetched[0], prefetched[1]};
            const uint16_t ws = prefetched_scales;
            const float ws0 = gemv_e4m3(ws & 255u) * 0x1p-13f;
            const float ws1 = gemv_e4m3(ws >> 8) * 0x1p-13f;
            if (kc + 128u < K) fetch(kc + 128u);
            float first_weighted[RB / 4];
#pragma unroll
            for (uint32_t group = 0; group < 2; ++group) {
                float raw[RB / 4];
#pragma unroll
                for (uint32_t q = 0; q < 4; ++q) {
                    // Every quartet sources the same two packed words.
                    const uint32_t packed0 = __shfl(current[group].x, q, 4);
                    const uint32_t packed1 = __shfl(current[group].y, q, 4);
                    const uint4 w0 = decode(packed0), w1 = decode(packed1);
                    const uint32_t kw = group * 8u + q * 2u;
#pragma unroll
                    for (uint32_t rb = 0; rb < RB / 4; ++rb) {
                        const uint32_t r = 4u * rb + qlane;
                        const float lo = dot8(activation[cur][r][kw], w0);
                        const float hi = dot8(activation[cur][r][kw + 1u], w1);
                        const float part = lo + hi;
                        if (q == 0) raw[rb] = part;
                        else raw[rb] += part;
                    }
                }
#pragma unroll
                for (uint32_t rb = 0; rb < RB / 4; ++rb) {
                    const uint32_t r = 4u * rb + qlane;
                    if (group == 0) first_weighted[rb] = raw[rb] * ws0;
                    else {
                        const float folded = first_weighted[rb] + raw[rb] * ws1;
                        acc[rb] += folded * activation_scale[cur][r];
                    }
                }
            }
            if (kc + 128u < K) stage(kc + 128u, cur ^ 1u);
            __syncthreads();
        }
        const uint32_t col = TOKEN && part_pitch && n >= N / 2 ? n - N / 2 + part_pitch : n;
#pragma unroll
        for (uint32_t rb = 0; rb < RB / 4; ++rb) {
            const uint32_t r = 4u * rb + qlane;
            if (r >= nr) continue;
            uint32_t u = __float_as_uint(acc[rb]);
            u += 0x7fffu + ((u >> 16) & 1u);
            out[(uint64_t)(r0 + r) * out_stride + col] = (uint16_t)(u >> 16);
        }
    }
}

}  // namespace r3_moe_gpu
