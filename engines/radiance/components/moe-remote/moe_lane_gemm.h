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

/* PREFILL MoE GEMM FOR FEW ROWS A EXPERT (RAD06), the helper's large lane. With the hot experts on the engine, a
 * helper's experts are cold: a handful of routed rows each per prefill step, where the grouped GEMM's 16-row
 * WMMA tiles are mostly padding. Here one block an (expert, 32 outputs): the expert's packed weight rows are
 * streamed once per 256-wide K chunk and each decoded weight is applied to up to 16 of the expert's rows,
 * whose activations sit dequantised in LDS. Same weight form, table, scales, sorted/off and part-pitch rule as
 * lane_moe_gemv; f32 accumulate, one bf16 narrow. */
constexpr uint32_t kRowsTileN = 32;       /* outputs a block: 8 waves x 4 */
constexpr uint32_t kRowsOut = 4;          /* outputs a wave */
constexpr uint32_t kRowsRB = 16;          /* rows a pass */
constexpr uint32_t kRowsKC = 256;         /* K a chunk */

template <bool TOKEN>
__global__ __launch_bounds__(256) void lane_moe_rows(
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
    if (n_base >= N || K % 8u) return;
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
            for (uint32_t i = threadIdx.x; i < nr * kRowsKC; i += blockDim.x) {
                if (i % kRowsKC >= kl) continue;
                const uint32_t r = i / kRowsKC, k = kc + i % kRowsKC;
                const uint32_t ar = arow[r];
                a[r][i % kRowsKC] = gemv_e4m3(aq[(uint64_t)ar * a_stride + k]) *
                                    as[(uint64_t)ar * (a_stride / 128u) + k / 128u];
            }
            __syncthreads();
            for (uint32_t d = lane; d < kl / 8u; d += warpSize) {
#pragma unroll
                for (uint32_t j = 0; j < kRowsOut; ++j) {
                    const uint32_t n = n_first + j;
                    if (n >= N) break;
                    const uint32_t w = reinterpret_cast<const uint32_t*>(codes + (uint64_t)n * (K / 2u))[kc / 8u + d];
                    const float s = gemv_e4m3(scales[(uint64_t)n * (K / 64u) + (kc + 8u * d) / 64u]);
                    float wv[8];
#pragma unroll
                    for (uint32_t b = 0; b < 4; ++b) {
                        const uint32_t byte = (w >> (8u * b)) & 0xFFu;
                        wv[b] = kGemvW4nl[byte & 15u] * s;
                        wv[b + 4] = kGemvW4nl[byte >> 4] * s;
                    }
#pragma unroll
                    for (uint32_t r = 0; r < kRowsRB; ++r) {
                        if (r >= nr) break;
                        const float* x = &a[r][8u * d];
                        acc[j][r] += x[0] * wv[0] + x[1] * wv[1] + x[2] * wv[2] + x[3] * wv[3] +
                                     x[4] * wv[4] + x[5] * wv[5] + x[6] * wv[6] + x[7] * wv[7];
                    }
                }
            }
        }
#pragma unroll
        for (uint32_t j = 0; j < kRowsOut; ++j) {
            const uint32_t n = n_first + j;
#pragma unroll
            for (uint32_t r = 0; r < kRowsRB; ++r) {
                if (r >= nr) break;
                float v = acc[j][r];
                for (uint32_t o = warpSize / 2; o; o >>= 1) v += __shfl_down(v, o);
                if (lane == 0 && n < N) {
                    uint32_t u = __float_as_uint(v * 0x1p-13f);
                    u += 0x7fffu + ((u >> 16) & 1u);
                    const uint32_t col = TOKEN && part_pitch && n >= N / 2 ? n - N / 2 + part_pitch : n;
                    out[(uint64_t)(r0 + r) * out_stride + col] = (uint16_t)(u >> 16);
                }
            }
        }
    }
}

}  // namespace r3_moe_gpu
