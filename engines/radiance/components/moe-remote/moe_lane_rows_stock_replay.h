#pragma once
// Historical inaccurate W4 kernel, isolated exclusively for --replay-sweep stock comparisons.
// Production W4 rows dispatch must never enter this file.
#include "moe_lane_gemm.h"
namespace r3_moe_gpu {
/* lane_moe_rows for w4nl weights (RAD23, the gfx1030 tail's large lane): fp16 operands and v_dot2_f32_f16, two products a
 * dword with an f32 accumulator. The chunk's activations are staged in LDS as fp16 (e4m3 times its scale), each output's 8
 * decoded weights a lane as fp16 pairs (table value times the e4m3 row scale, at most 22 x 448, inside fp16), so the
 * inner loop is one 16-byte LDS read a row and four dot2 an output, against eight f32 FMAs and two reads. The instruction
 * is written as asm: __builtin_amdgcn_fdot2 gives wrong sums on gfx1030 (libr2 r3_mma16.h). RB rows a pass, OUTW outputs a
 * wave (a block 8 x OUTW outputs): gate/up 16 x 4, down 12 x 2 measured best on a 6900 XT with the live tail's rows-an-expert
 * mix (tools/lab/radiance/gfx1030/rowsk.hip: gate/up 2563 -> 1911 us, down 1994 -> 1568 us a request; output against the
 * f32 kernel rel_l2 8.7e-4, the bf16 narrow). The activation tile is double-buffered: the next chunk is staged while this one
 * is computed, one barrier a chunk instead of two (gate/up 1924 -> 1765 us, down 1568 -> 1406 us; bit-identical). Not
 * bit-exact with the f32 form. */
constexpr uint32_t kRowsGuRB = 16, kRowsGuOut = 4;   /* gate/up */
constexpr uint32_t kRowsDnRB = 12, kRowsDnOut = 2;   /* down */
template <bool TOKEN, uint32_t RB, uint32_t OUTW>
__global__ __launch_bounds__(256) void lane_moe_rows_w4_stock_replay(
    const uint8_t* aq, const float* as, const int32_t* sorted, const int32_t* off, uint32_t top_k,
    void* const* code0, void* const* scale0, void* const* code1, void* const* scale1,
    uint32_t n0, uint32_t n1, uint32_t k0, uint32_t k1, uint32_t a_stride, uint32_t out_stride, uint16_t* out,
    uint32_t part_pitch) {
    const uint32_t e = blockIdx.x;
    const uint32_t rs = (uint32_t)off[e], re = (uint32_t)off[e + 1];
    if (rs >= re) return;
    const uint32_t cls = e & 1u, ix = e >> 1;
    const uint32_t N = cls ? n1 : n0, K = cls ? k1 : k0;
    const uint32_t n_base = blockIdx.y * (8u * OUTW);
    if (n_base >= N || K % 8u) return;
    const uint8_t* codes = static_cast<const uint8_t*>(cls ? code1[ix] : code0[ix]);
    const uint8_t* scales = static_cast<const uint8_t*>(cls ? scale1[ix] : scale0[ix]);
    if (!codes || !scales) return;
    __shared__ uint4 a[2][RB][kRowsKC / 8];   /* 8 fp16 activations a uint4 */
    __shared__ uint32_t arow[RB];
    __shared__ float lut[16];
    if (threadIdx.x < 16) lut[threadIdx.x] = kGemvW4nl[threadIdx.x];
    const uint32_t lane = threadIdx.x % warpSize, wave = threadIdx.x / warpSize;
    const uint32_t n_first = n_base + wave * OUTW;
    for (uint32_t r0 = rs; r0 < re; r0 += RB) {
        const uint32_t nr = re - r0 < RB ? re - r0 : RB;
        if (threadIdx.x < nr) arow[threadIdx.x] = TOKEN ? (uint32_t)sorted[r0 + threadIdx.x] / top_k : r0 + threadIdx.x;
        float acc[OUTW][RB];
#pragma unroll
        for (uint32_t j = 0; j < OUTW; ++j)
#pragma unroll
            for (uint32_t r = 0; r < RB; ++r) acc[j][r] = 0.0f;
        uint32_t pw[OUTW]; uint8_t ps[OUTW];
        auto fetch = [&](uint32_t kcn) {
            const uint32_t kln = K - kcn < kRowsKC ? K - kcn : kRowsKC;
#pragma unroll
            for (uint32_t j = 0; j < OUTW; ++j) {
                const uint32_t n = n_first + j;
                const bool ok = n < N && lane < kln / 8u;
                pw[j] = ok ? reinterpret_cast<const uint32_t*>(codes + (uint64_t)n * (K / 2u))[kcn / 8u + lane] : 0u;
                ps[j] = ok ? scales[(uint64_t)n * (K / 64u) + (kcn + 8u * lane) / 64u] : (uint8_t)0;
            }
        };
        fetch(0);
        auto stage = [&](uint32_t kcs, uint32_t buf) {
            const uint32_t kls = K - kcs < kRowsKC ? K - kcs : kRowsKC;
            {
                const uint32_t pieces = kRowsKC / 16u;
                for (uint32_t idx = threadIdx.x; idx < RB * pieces; idx += blockDim.x) {
                const uint32_t r = idx / pieces, p16 = idx % pieces;
                if (r < nr && 16u * p16 < kls) {
                    const uint32_t ar = arow[r], k = kcs + 16u * p16;
                    const uint4 q = *reinterpret_cast<const uint4*>(aq + (uint64_t)ar * a_stride + k);
                    const float sc = as[(uint64_t)ar * (a_stride / 128u) + k / 128u];
                    const uint32_t qw[4] = {q.x, q.y, q.z, q.w};
                    uint32_t hw[8];
#pragma unroll
                    for (uint32_t w = 0; w < 4; ++w) {
                        const _Float16 h0 = (_Float16)(gemv_e4m3(qw[w] & 0xFFu) * sc), h1 = (_Float16)(gemv_e4m3((qw[w] >> 8) & 0xFFu) * sc);
                        const _Float16 h2 = (_Float16)(gemv_e4m3((qw[w] >> 16) & 0xFFu) * sc), h3 = (_Float16)(gemv_e4m3(qw[w] >> 24) * sc);
                        hw[2 * w] = (uint32_t)__builtin_bit_cast(uint16_t, h0) | ((uint32_t)__builtin_bit_cast(uint16_t, h1) << 16);
                        hw[2 * w + 1] = (uint32_t)__builtin_bit_cast(uint16_t, h2) | ((uint32_t)__builtin_bit_cast(uint16_t, h3) << 16);
                    }
                    a[buf][r][2 * p16] = make_uint4(hw[0], hw[1], hw[2], hw[3]);
                    a[buf][r][2 * p16 + 1] = make_uint4(hw[4], hw[5], hw[6], hw[7]);
                }
                }
            }
        };
        __syncthreads();   /* the previous pass's reads of a[] are done */
        stage(0, 0);
        __syncthreads();
        uint32_t cur = 0;
        for (uint32_t kc = 0; kc < K; kc += kRowsKC, cur ^= 1u) {
            const uint32_t kl = K - kc < kRowsKC ? K - kc : kRowsKC;
            uint32_t cw[OUTW]; uint8_t cs[OUTW];
#pragma unroll
            for (uint32_t j = 0; j < OUTW; ++j) { cw[j] = pw[j]; cs[j] = ps[j]; }
            if (kc + kRowsKC < K) fetch(kc + kRowsKC);
            if (const uint32_t d = lane; d < kl / 8u) {
                uint32_t wv[OUTW][4];   /* the 8 decoded weights of each output as fp16 pairs, in K order */
#pragma unroll
                for (uint32_t j = 0; j < OUTW; ++j) {
                    const uint32_t n = n_first + j;
                    const float s = n < N ? gemv_e4m3(cs[j]) : 0.0f;
                    const uint32_t w = cw[j];
                    /* K order 8d+0..7: low nibbles of bytes 0..3 are K 0..3, high nibbles K 4..7 */
                    const _Float16 k0 = (_Float16)(lut[w & 15u] * s), k1 = (_Float16)(lut[(w >> 8) & 15u] * s);
                    const _Float16 k2 = (_Float16)(lut[(w >> 16) & 15u] * s), k3 = (_Float16)(lut[(w >> 24) & 15u] * s);
                    const _Float16 k4 = (_Float16)(lut[(w >> 4) & 15u] * s), k5 = (_Float16)(lut[(w >> 12) & 15u] * s);
                    const _Float16 k6 = (_Float16)(lut[(w >> 20) & 15u] * s), k7 = (_Float16)(lut[(w >> 28) & 15u] * s);
                    auto pk = [](_Float16 x, _Float16 y) { return (uint32_t)__builtin_bit_cast(uint16_t, x) | ((uint32_t)__builtin_bit_cast(uint16_t, y) << 16); };
                    wv[j][0] = pk(k0, k1); wv[j][1] = pk(k2, k3); wv[j][2] = pk(k4, k5); wv[j][3] = pk(k6, k7);
                }
#pragma unroll
                for (uint32_t r = 0; r < RB; ++r) {
                    if (r >= nr) continue;
                    const uint4 x = a[cur][r][d];
#pragma unroll
                    for (uint32_t j = 0; j < OUTW; ++j) {
                        float s = acc[j][r];
                        asm("v_dot2_f32_f16 %0, %1, %2, %0" : "+v"(s) : "v"(x.x), "v"(wv[j][0]));
                        asm("v_dot2_f32_f16 %0, %1, %2, %0" : "+v"(s) : "v"(x.y), "v"(wv[j][1]));
                        asm("v_dot2_f32_f16 %0, %1, %2, %0" : "+v"(s) : "v"(x.z), "v"(wv[j][2]));
                        asm("v_dot2_f32_f16 %0, %1, %2, %0" : "+v"(s) : "v"(x.w), "v"(wv[j][3]));
                        acc[j][r] = s;
                    }
                }
            }
            if (kc + kRowsKC < K) stage(kc + kRowsKC, cur ^ 1u);
            __syncthreads();
        }
#pragma unroll
        for (uint32_t j = 0; j < OUTW; ++j) {
            const uint32_t n = n_first + j;
#pragma unroll
            for (uint32_t r = 0; r < RB; ++r) {
                if (r >= nr) continue;
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



} // namespace r3_moe_gpu
