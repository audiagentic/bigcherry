// bc_moe_gemv.hip -- moe_gemm_q (fp8a8) at decode-sized steps on gfx1201 (RAD15): the moe-remote worker's packed-f16 GEMV
// (lane_moe_gemv_h2, RAD14) on libr4d's operands. libr4d's decode form runs the fp8 WMMA tile over a 16-row tile per routed
// run, most of it masked at one or two rows a run; here one block takes a (routed row pair, 64 outputs) and each wave one
// output, the dot product on v_dot2_f32_f16. An E4M3 code is exact in f16 and so is the product of two, so nothing rounds
// before the f32 accumulation; the activation's per-128 scale and the weight's per-128x128 tile scale are applied to each
// sixteen-element partial sum. The 35B's gate/up call (rad-kbench, R9700): 28/56/110/213/389 us against libr4d's
// 34/80/158/267/455 at M = 1/2/4/8/16. The down call (sorted activation) is declined: libr4d's own down decode kernel wins.
//
// The weight is libr4d's own stored form (r4d_rad_layout_moe_fp8: the checkpoint's row-major bytes), two classes by expert
// parity, each a pointer table (zero leading stride) or a stacked plane. A geometry this kernel does not take is declined
// with RAD_E_UNSUPPORTED and the row hands the call to libr4d.
#include "rad_abi.h"

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

constexpr uint32_t kTile = 64;      /* outputs a block */
constexpr uint32_t kMaxK = 2560;    /* the activation row staged in LDS */
constexpr uint32_t kRows = 2;       /* routed rows of one expert a block, each weight decoded once for both */

struct Cls {
    const void* w;                  /* pointer table, or the stacked plane's base */
    const void* s;
    uint64_t w_s0, s_s0;            /* bytes an entry of a stacked plane; 0 marks a pointer table */
    uint32_t n, k, w_ld, s_ld;      /* w_ld bytes a weight row, s_ld bf16 a scale row */
};

typedef _Float16 h2v __attribute__((ext_vector_type(2)));
__device__ __forceinline__ h2v e4m3x2_to_h2(uint32_t x) {   /* codes in bits 0..7 and 16..23 */
    const uint32_t h = ((x & 0x007F007Fu) << 7) | ((x & 0x00800080u) << 8);
    return __builtin_bit_cast(h2v, h) * h2v{(_Float16)256.0f, (_Float16)256.0f};
}
__device__ __forceinline__ const uint8_t* entry(const void* base, uint64_t s0, uint32_t ix) {
    return s0 ? static_cast<const uint8_t*>(base) + (uint64_t)ix * s0
              : static_cast<const uint8_t*>(static_cast<const void* const*>(base)[ix]);
}

/* TOKEN: the activation row is the routed slot's token (sorted[r] / top_k); otherwise the sorted row itself. Rows past
 * off[n_expert] (dropped slots) are zeroed. The output is `parts` stacked parts of N / parts columns, as is every weight
 * (n / parts rows a part); a class's part narrower than the output's fills the part's leading columns and the rest are
 * zero (libr4d's rule, r4d_moe.hip). */
template <bool TOKEN>
__global__ __launch_bounds__(256) void moe_gemv_fp8_h2(
    const uint8_t* aq, const float* as, const int32_t* sorted, const int32_t* off, Cls c0, Cls c1, uint16_t* y,
    uint32_t N, uint32_t parts, uint32_t n_expert, uint32_t top_k, uint32_t a_ld, uint32_t as_ld, uint32_t y_ld) {
    const uint32_t r = blockIdx.x;
    const uint32_t n_base = blockIdx.y * kTile;
    if (r >= (uint32_t)off[n_expert]) {
        for (uint32_t j = threadIdx.x; j < kTile && n_base + j < N; j += blockDim.x) y[(uint64_t)r * y_ld + n_base + j] = 0;
        return;
    }
    __shared__ uint32_t a[kRows][kMaxK / 2];   /* K half values as half2 pairs, in two planes, a row */
    __shared__ float asc[kRows][kMaxK / 128u];
    __shared__ uint32_t sh_e, sh_first;
    if (threadIdx.x == 0) {
        uint32_t lo = 0, hi = n_expert;
        while (hi - lo > 1) { const uint32_t mid = (lo + hi) / 2; if ((uint32_t)off[mid] <= r) lo = mid; else hi = mid; }
        sh_e = lo;
        sh_first = (uint32_t)off[lo];
    }
    __syncthreads();
    const uint32_t e = sh_e, ix = e >> 1;
    if ((r - sh_first) % kRows) return;        /* a follower row: its leader computes it */
    const uint32_t nr = (uint32_t)off[e + 1] - r < kRows ? (uint32_t)off[e + 1] - r : kRows;
    const Cls& c = (e & 1u) ? c1 : c0;
    const uint32_t K = c.k, fy = N / parts, fc = c.n / parts;
    if (n_base >= N) return;
    const uint32_t k16 = K / 16u;
    /* element k = 16 d + 4 q + j sits in plane (j & 1), half2 index (d * 4 + q), lane (j >> 1) */
    for (uint32_t m = 0; m < nr; ++m) {
        const uint32_t arow = TOKEN ? (uint32_t)sorted[r + m] / top_k : r + m;
        const uint8_t* ar = aq + (uint64_t)arow * a_ld;
        const float* asr = as + (uint64_t)arow * as_ld;
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

    const uint8_t* codes = entry(c.w, c.w_s0, ix);
    const uint16_t* scales = reinterpret_cast<const uint16_t*>(entry(c.s, c.s_s0, ix));
    const uint32_t lane = threadIdx.x % warpSize, wave = threadIdx.x / warpSize, waves = blockDim.x / warpSize;
    constexpr uint32_t U = 4;
    for (uint32_t j = wave; j < kTile; j += waves) {
        const uint32_t n = n_base + j;
        if (n >= N) break;
        const uint32_t part = n / fy, jp = n - part * fy;
        if (jp >= fc) {                               /* past this class's part: zero */
            if (lane == 0)
                for (uint32_t m = 0; m < nr; ++m) y[(uint64_t)(r + m) * y_ld + n] = 0;
            continue;
        }
        const uint32_t wn = part * fc + jp;
        const uint4* crow = reinterpret_cast<const uint4*>(codes + (uint64_t)wn * c.w_ld);
        const uint16_t* srow = scales + (uint64_t)(wn / 128u) * c.s_ld;
        float acc[kRows];
#pragma unroll
        for (uint32_t m = 0; m < kRows; ++m) acc[m] = 0.0f;
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
                for (uint32_t m = 0; m < kRows; ++m) {
                    if (m >= nr) break;
                    const uint4 xe = reinterpret_cast<const uint4*>(a[m])[d], xo = reinterpret_cast<const uint4*>(a[m])[k16 + d];
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
        for (uint32_t m = 0; m < kRows; ++m) {
            if (m >= nr) break;
            float x = acc[m];
            for (uint32_t o = warpSize / 2; o; o >>= 1) x += __shfl_down(x, o);
            if (lane == 0) {
                uint32_t u = __float_as_uint(x);
                u += 0x7fffu + ((u >> 16) & 1u);
                y[(uint64_t)(r + m) * y_ld + n] = (uint16_t)(u >> 16);
            }
        }
    }
}

long long geti(const RadArgs* a, const char* key, long long dflt) {
    long long v = dflt;
    rad_args_geti(a, key, &v);
    return v;
}
/* A rank-2 row-major operand: its rows and row pitch (elements), or -1. */
long long rows_pitch(const RadTensor* t, long long cols, long long* rows) {
    if (!t || !t->data || t->rank != 2 || t->stride[1] != 1 || t->shape[1] < cols) return -1;
    *rows = t->shape[0];
    return t->stride[0];
}
/* One class's tables, as libr4d's wtab_of reads them: rank 3 [entries, rows, cols], a zero leading stride a pointer table. */
bool cls_of(const RadTensor* w, const RadTensor* s, long long cnt, long long n, long long k, Cls* c) {
    if (!w || !w->data || w->rank != 3 || w->dtype != RAD_F8E4M3 || w->stride[2] != 1) return false;
    if (!s || !s->data || s->rank != 3 || s->dtype != RAD_BF16 || s->stride[2] != 1) return false;
    if (w->shape[0] < cnt || w->shape[1] < n || w->shape[2] < k) return false;
    if (s->shape[0] < cnt || s->shape[1] < (n + 127) / 128 || s->shape[2] < k / 128) return false;
    /* sixteen-byte weight loads: a row's pitch and a stacked entry whole sixteen-byte pieces */
    if (w->stride[1] % 16 || w->stride[0] % 16) return false;
    c->w = w->data;
    c->s = s->data;
    c->w_s0 = (uint64_t)w->stride[0];
    c->s_s0 = (uint64_t)s->stride[0] * 2u;
    c->n = (uint32_t)n;
    c->k = (uint32_t)k;
    c->w_ld = (uint32_t)w->stride[1];
    c->s_ld = (uint32_t)s->stride[1];
    return true;
}

}  // namespace

extern "C" int bc_moe_gemm_fp8_decode(const RadArgs* a, RadStream stream) {
    /* On by default (BC_GFX12_MOE_GEMV=0 turns it off): 1-stream greedy output is not identical to libr4d's (a near-tie
     * flips ~45 tokens in; f32 accumulation order differs from WMMA's), measured at 99.06% top-1 agreement / KL mean
     * 3.0e-3 / p99 5.7e-2 against libr4d (KL mode is prefill-only and cannot reach a decode-only row). Single-model
     * (35B) evidence, accepted by the owner 2026-10-21 after two genuine architecture-mismatch dead ends looking for
     * a second fp8a8-MoE model radiance's architecture plugins actually support (RAD15). */
    static const bool on = [] {
        const char* v = std::getenv("BC_GFX12_MOE_GEMV");
        const bool o = !v || std::strcmp(v, "0");
        if (o) std::fprintf(stderr, "libbcgfx12: BC_GFX12_MOE_GEMV on\n");
        return o;
    }();
    if (!on || !a || a->n_t < 9) return RAD_E_UNSUPPORTED;
    const RadTensor *Aq = &a->t[0], *As = &a->t[1], *sorted = &a->t[4], *off = &a->t[5], *Y = &a->t[8];
    const long long n_expert = geti(a, "n_expert", 0), top_k = geti(a, "top_k", 0), parts = geti(a, "parts", 1);
    const long long N0 = geti(a, "N", 0), K0 = geti(a, "K", 0);
    const long long N1 = geti(a, "N_odd", N0), K1 = geti(a, "K_odd", K0);
    if (n_expert <= 0 || top_k <= 0 || parts <= 0) return RAD_E_UNSUPPORTED;
    if (N0 <= 0 || N1 <= 0 || K0 <= 0 || K1 <= 0 || N0 % (64 * parts) || N1 % (64 * parts) || K0 % 128 || K1 % 128)
        return RAD_E_UNSUPPORTED;
    if (K0 > kMaxK || K1 > kMaxK) return RAD_E_UNSUPPORTED;
    const long long N = N0 > N1 ? N0 : N1, K = K0 > K1 ? K0 : K1;
    if (Aq->dtype != RAD_F8E4M3 || As->dtype != RAD_F32 || Y->dtype != RAD_BF16) return RAD_E_UNSUPPORTED;
    if (sorted->dtype != RAD_I32 || off->dtype != RAD_I32 || !sorted->data || !off->data) return RAD_E_UNSUPPORTED;
    long long T = 0, M = 0, Ms = 0;
    const long long y_ld = rows_pitch(Y, N, &T), a_ld = rows_pitch(Aq, K, &M), as_ld = rows_pitch(As, K / 128, &Ms);
    if (y_ld < 0 || a_ld < 0 || as_ld < 0 || T <= 0 || Ms < M) return RAD_E_UNSUPPORTED;
    if (rad_tensor_numel(sorted) < T || rad_tensor_numel(off) < n_expert + 1) return RAD_E_UNSUPPORTED;
    const char* ord = rad_args_gets(a, "a_order");
    const bool a_sorted = ord && !std::strcmp(ord, "sorted");
    if (ord && !a_sorted && std::strcmp(ord, "token")) return RAD_E_UNSUPPORTED;
    if (M < (a_sorted ? T : (T + top_k - 1) / top_k)) return RAD_E_UNSUPPORTED;
    /* The down projection (sorted activation) stays libr4d's: its own decode kernel (moe_down_dec_kernel) times 3-25%
     * faster than this one at M = 1..16 on the 35B (rad-kbench, R9700); the gate/up call is 15-31% faster here. */
    if (a_sorted) return RAD_E_UNSUPPORTED;
    /* libr4d's decode band (r4d_moe_decode_band); a larger step is a GEMM, not this kernel's shape */
    if (2 * T > 3 * n_expert) return RAD_E_UNSUPPORTED;
    Cls c0{}, c1{};
    if (!cls_of(&a->t[2], &a->t[3], (n_expert + 1) / 2, N0, K0, &c0)) return RAD_E_UNSUPPORTED;
    if (!cls_of(&a->t[6], &a->t[7], n_expert / 2, N1, K1, &c1)) return RAD_E_UNSUPPORTED;
    const dim3 grid((unsigned)T, (unsigned)((N + kTile - 1) / kTile));
    if (a_sorted)
        hipLaunchKernelGGL(moe_gemv_fp8_h2<false>, grid, dim3(256), 0, (hipStream_t)stream, (const uint8_t*)Aq->data,
                           (const float*)As->data, (const int32_t*)sorted->data, (const int32_t*)off->data, c0, c1,
                           (uint16_t*)Y->data, (uint32_t)N, (uint32_t)parts, (uint32_t)n_expert, (uint32_t)top_k, (uint32_t)a_ld,
                           (uint32_t)as_ld, (uint32_t)y_ld);
    else
        hipLaunchKernelGGL(moe_gemv_fp8_h2<true>, grid, dim3(256), 0, (hipStream_t)stream, (const uint8_t*)Aq->data,
                           (const float*)As->data, (const int32_t*)sorted->data, (const int32_t*)off->data, c0, c1,
                           (uint16_t*)Y->data, (uint32_t)N, (uint32_t)parts, (uint32_t)n_expert, (uint32_t)top_k, (uint32_t)a_ld,
                           (uint32_t)as_ld, (uint32_t)y_ld);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_UNSUPPORTED;
}
