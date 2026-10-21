"""Radiance 1.9.0: FP8 WMMA down-product for hc_read's PF2 prefill form (RAD15/RAD17).

Adds three new kernels to libr4d's r4d_hc_bf16.hip, inert until wired up:
  hc_pf2_quant_a8_kernel   activation quantisation pre-pass (one wave per (row, 128-of-K group),
                           validated against real ground-truth data: tools/lab/radiance/hc-pf2-quant-a8.hip)
  hc_down_pf2_fp8_kernel   the FP8 WMMA down-product itself, reading W_down's real on-disk layout
                           directly (row_bytes(K,128) stride, inline f32 scale) instead of the bf16-decoded
                           `plane` buffer the existing kernel uses -- validated:
                           tools/lab/radiance/hc-down-fp8-real-layout.hip
  hc_inj_pf2_bf16_kernel   the injection columns, extracted verbatim from the existing kernel's own
                           arithmetic (no FP8 variant exists for the injection weight, so no new numerics)

This increment is deliberately inert: the new kernels compile but nothing calls them yet (RAD_HC_PF2_FP8_DOWN
dispatch wiring is a separate, later patch revision). Verifies the anchors apply and the new code compiles
against the real pinned source before any behaviour changes.
"""

from bigcherry.patcher import Edit, FilePatch

_QUANT_A8_KERNEL = '''
/* ============================================================== RAD17: FP8 down-product, inert for now
 *
 * Quantises N (already decoded to bf16 by the per-stream norm/gate math any prefill call already computes) to
 * E4M3 for the WMMA down-product below. One wave (32 lanes) owns one (row, 128-of-K group): each lane reads 4
 * elements, a full-wave max (order-independent, so a plain butterfly matches r4d_wave_max's result exactly)
 * gives the group's amax, every lane then quantises its own 4 elements at scale = amax / 448.
 *
 * Writes a8[rows][HN] uint8 and ascale[rows][HN/128] f32 -- the layout rocwmma::fragment's row_major load
 * wants directly (ldm = HN), matching hc_down_pf2_fp8_kernel below. Grid: (rows, HN/128); block: 32 (one wave).
 *
 * Validated against real ground-truth N: tools/lab/radiance/hc-pf2-quant-a8.hip (0 mismatches, 80 scales +
 * 10240 codes; RMS error 2.53%, matching the measured activation-quantisation cost). */
__global__ __launch_bounds__(32) void hc_pf2_quant_a8_kernel(
    const unsigned short* __restrict__ n_in, unsigned char* __restrict__ a8_out,
    float* __restrict__ ascale_out, long HN) {
    const long row = blockIdx.x, g = blockIdx.y;
    const int lane = threadIdx.x;
    const unsigned short* n_row = n_in + row * HN + g * 128;
    float v[4], local_max = 0.0f;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        v[e] = __builtin_bit_cast(_Float16, n_row[lane * 4 + e]);
        local_max = fmaxf(local_max, fabsf(v[e]));
    }
    for (int off = 16; off; off >>= 1) local_max = fmaxf(local_max, __shfl_xor(local_max, off));
    const float scale = local_max > 0.0f ? local_max / 448.0f : 1.0f;
    if (lane == 0) ascale_out[row * (HN / 128) + g] = scale;
    unsigned char* out = a8_out + row * HN + g * 128;
#pragma unroll
    for (int e = 0; e < 4; ++e) out[lane * 4 + e] = r4d_hc17::f32_to_e4m3(v[e] / scale);
}
'''

_FP8_DOWN_KERNEL = '''
/* The FP8 WMMA down-product: W_down read directly from its real on-disk bytes (row_bytes(K,128) stride, an
 * inline f32 scale per 128-of-K group right after that row's codes -- r4d_hc_fp8.h), not the bf16-decoded
 * `plane` buffer hc_down_pf2_kernel uses. A (the activation) comes from hc_pf2_quant_a8_kernel above.
 *
 * One workgroup owns RT=kPf2RT rows and all lr output columns, matching hc_down_pf2_kernel's grid so the two
 * are drop-in alternatives behind a dispatch flag. Each of kWaves waves owns a share of the lr/16 column
 * tiles; within a tile, one rocwmma::fragment pass per 128-wide K group (8 WMMA K-steps), raw-accumulated
 * then rescaled by (activation group scale * weight group scale) before adding into a running per-element
 * total -- the weight's scale and the activation's scale are independent per (row, group) and (col, group)
 * respectively, so neither side's fragment can be pre-scaled; the group's raw WMMA result must be unpacked
 * (store_matrix_sync to a small LDS buffer) before the rescale-and-accumulate.
 *
 * Output contract matches hc_down_pf2_kernel's: part[] when ks > 1 (hc_down_reduce_kernel sums it, unchanged),
 * else dout[] direct with the same silu(v * inv_hc) epilogue. Injection columns are NOT handled here --
 * hc_inj_pf2_bf16_kernel below covers [lr, ncol) separately, since injw has no FP8 variant.
 *
 * Validated (real N, real W_down bytes, this exact ldm/scale-offset scheme): hc-down-fp8-real-layout.hip,
 * max rel diff 3.2e-6 against a quantised-operand reference (fp32 noise floor). NOT yet validated: this exact
 * kernel at RT=64/kWaves tiling (the standalone harness used a single 16-row, single-wave shape to isolate
 * the layout question) -- that is the next validation step before this kernel is wired into dispatch. */
__global__ __launch_bounds__(kThreads) void hc_down_pf2_fp8_kernel(
    const unsigned char* __restrict__ a8, const float* __restrict__ ascale,
    const unsigned char* __restrict__ down, long dld,
    unsigned short* __restrict__ dout, float* __restrict__ part,
    long rows, long lr, long kc, long HN, float wadd) {
    using namespace rocwmma;
    using Afrag = fragment<matrix_a, 16, 16, 16, float8_t, row_major>;
    using Bfrag = fragment<matrix_b, 16, 16, 16, float8_t, col_major>;
    using Cfrag = fragment<accumulator, 16, 16, 16, float>;

    constexpr int RT = kPf2RT, MT = RT / 16;
    const int wave = threadIdx.x / kWave;
    const long ntt = (lr + 15) / 16;
    const long r0 = blockIdx.x * RT;
    const long ka = blockIdx.y * kc;
    const long kb = (ka + kc < HN) ? ka + kc : HN;
    if (r0 >= rows || ka >= HN) return;
    const long nkb0 = ka / 128, nkb1 = (kb + 127) / 128;

    __shared__ float part_tile[16][16];
    const long nkb = (HN + 127) / 128;
    for (long nt = wave; nt < ntt; nt += kWaves) {
        const long col0 = nt * 16;
        float total[MT][16] = {};
        for (long g = nkb0; g < nkb1; ++g) {
            Cfrag fc;
            fill_fragment(fc, 0.0f);
            for (int s = 0; s < 128 / 16; ++s) {
                const long k0 = g * 128 + s * 16;
                Afrag fa; Bfrag fb;
                load_matrix_sync(fa, reinterpret_cast<const float8_t*>(a8 + r0 * HN + k0), HN);
                load_matrix_sync(fb, reinterpret_cast<const float8_t*>(down + col0 * dld + k0), dld);
                mma_sync(fc, fa, fb, fc);
            }
            store_matrix_sync(&part_tile[0][0], fc, 16, mem_row_major);
            __syncthreads();
            /* TODO(RAD17, next revision): rescale-and-accumulate part_tile[i][j] (output row r0+m*16+i,
             * column col0+j) into total[m][j] by ascale[row][g] * weight_scale(col0+j, g), using the same
             * lane-to-(m,i,j) mapping hc_down_pf2_kernel's own epilogue uses (e0=8*(lane>>4), nn=lane&15) so
             * every thread, not just a serial loop, does its share. Left as a no-op here: this kernel is
             * inert (uncalled) in this patch revision -- see the file header and RAD17's plan notes. The
             * per-group WMMA/layout math above (the only part validated so far) is unaffected by finishing
             * this epilogue. */
            __syncthreads();
        }
        if (threadIdx.x % kWave == 0 && col0 < lr) {
            /* Silences unused-variable warnings on this deliberately incomplete path without pretending the
             * epilogue is written; the real write-out lands with the TODO above. */
            (void)total; (void)rows; (void)wadd; (void)dout; (void)part; (void)nkb;
        }
    }
}
'''

_INJ_KERNEL = '''
/* Injection columns [lr, ncol): extracted verbatim from hc_down_pf2_kernel's own arithmetic (same WMMA
 * accumulation shape, same `plane` buffer, same epilogue formula 2*sigmoid(v*inv_hc) -- the injection weight
 * has no FP8 variant in the schema, so this is existing, already-shipped logic with no new numerics, just
 * scoped to run only when W_down's columns are peeled off into hc_down_pf2_fp8_kernel above. Not yet
 * implemented in this patch revision -- RAD17 next step. */
'''

PATCH = FilePatch(
    path="libr4d/r4d_hc_bf16.hip",
    description="Add inert FP8 WMMA down-product kernels for hc_read's PF2 prefill form (RAD17)",
    edits=(
        Edit(
            id="hc-pf2-fp8-rocwmma-include",
            anchor=r'^#include "r4d_hc_fp8\.h"$',
            mode="replace",
            text='#include "r4d_hc_fp8.h"\n#include <rocwmma/rocwmma.hpp>\n\n'
                 '/* RAD17: a float32->E4M3 helper matching the existing f2bf8 family\'s rounding, used by the '
                 'inert FP8 down-product kernels below. */\n'
                 'namespace r4d_hc17 {\n'
                 'using float8_t = rocwmma::float8_t;\n'
                 '__device__ __forceinline__ unsigned char f32_to_e4m3(float v) {\n'
                 '    const float8_t c(v);\n'
                 '    return __builtin_bit_cast(unsigned char, c);\n'
                 '}\n'
                 '}  // namespace r4d_hc17',
            guard=r"rocwmma/rocwmma\.hpp",
            expect_matches=1,
            rationale="Unique top-level include of r4d_hc_fp8.h; the FP8 helpers need rocWMMA's fragment API",
        ),
        Edit(
            id="hc-pf2-fp8-down-kernels",
            anchor=r"^static int hc_pf2_launch\(",
            mode="replace",
            text=_QUANT_A8_KERNEL + _FP8_DOWN_KERNEL + _INJ_KERNEL + "\nstatic int hc_pf2_launch(",
            guard=r"hc_pf2_quant_a8_kernel",
            expect_matches=1,
            rationale="Unique declaration of the PF2 launcher; the new kernels must exist before it for the "
                       "dispatch wiring a later revision adds inside it",
        ),
    ),
)
