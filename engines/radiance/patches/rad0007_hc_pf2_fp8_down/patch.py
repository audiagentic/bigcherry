"""Radiance 1.9.0: FP8 WMMA down-product for hc_read's PF2 prefill form (RAD15/RAD17).

Adds three new kernels to libr4d's r4d_hc_bf16.hip, inert until wired up:
  hc_pf2_quant_a8_kernel   builds N inline AND quantises it to E4M3 in one pass, reading the SAME real
                           inputs (h, the published per-stream norm scale, the gamma weight w) and running
                           the IDENTICAL formula the real hc_down_pf2_kernel's stageA already runs on
                           hardware -- there is no materialized bf16 N array anywhere in the real pipeline to
                           read from instead (an earlier revision of this kernel wrongly assumed one existed;
                           found by reading hc_down_pf2_kernel's fetchA/stageA directly). Quantisation math
                           validated against real ground-truth N: tools/lab/radiance/hc-pf2-quant-a8.hip.
  hc_down_pf2_fp8_kernel   the FP8 WMMA down-product itself, reading W_down's real on-disk layout
                           directly (row_bytes(K,128) stride, inline f32 scale) instead of the bf16-decoded
                           `plane` buffer the existing kernel uses. Output contract (part[]/dout write) is
                           transcribed verbatim from the real hc_down_pf2_kernel/hc_down_reduce_kernel.
                           ks==1 path validated end to end at real RT=64/lr=320/kWaves=8 scale:
                           tools/lab/radiance/hc-down-fp8-fullscale.hip.
  hc_inj_pf2_bf16_kernel   the injection columns, extracted verbatim from the existing kernel's own
                           arithmetic (no FP8 variant exists for the injection weight, so no new numerics)

Dispatch is wired (RAD_HC_PF2_FP8_DOWN=1, default off) into hc_pf2_launch's down-product call site,
including the split-K case (every real PF2-eligible shape has ks>1; a ks==1-only guard would have been dead
code for every real call -- found by running this on hardware). Ran end to end on a real gfx1201 device via
r4d_selftest --perfhc: no crash, flag-off path byte-identical to baseline, but flag-on currently measures
2.7x SLOWER than the existing bf16 path and has not yet been checked against a quality metric (byte-identity
fails as expected for a lossy FP8 path; no RMS/KL-style check has been built). Not yet a performance win --
see SUMMARY.md for the measured numbers and what's still needed.
"""

from bigcherry.patcher import Edit, FilePatch

_QUANT_A8_KERNEL = '''
/* ============================================================== RAD17: FP8 down-product, inert for now
 *
 * Builds N inline AND quantises it to E4M3 in the same pass -- there is no materialized bf16 N array
 * anywhere in the real pipeline to read from (hc_down_pf2_kernel builds it incrementally, tile by tile, into
 * a tiny double-buffered LDS window, and never stores the whole thing -- confirmed by reading that kernel's
 * fetchA/stageA directly for this revision; an earlier version of this kernel wrongly assumed such a buffer
 * existed). So this kernel reads the SAME real inputs stageA does -- `h` (raw bf16), the per-(row,stream)
 * norm scale hc_pf2_prep_kernel already publishes to `scalein`, and the per-channel gamma `w` -- and applies
 * the IDENTICAL formula stageA already runs on real hardware every call: v = bf2f(h) * scale * (w + wadd).
 * This is not new numerics -- it is a transcription of already-shipped, already r4d_selftest-checked
 * arithmetic (see this file's own doc comment above hc_down_pf2_kernel) -- the only genuinely new piece is
 * the E4M3 quantisation that follows, which is unchanged from the previous revision and independently
 * validated against real data.
 *
 * One wave (32 lanes) owns one (row, 128-of-K group); 128 divides n (2560) exactly, so a group never
 * straddles a stream boundary and the stream index c = k0/n is wave-uniform. Each lane computes 4 elements,
 * a full-wave max (order-independent, so a plain butterfly matches r4d_wave_max's result exactly) gives the
 * group's amax, every lane then quantises its own 4 elements at scale = amax / 448.
 *
 * Writes a8[rows][HN] uint8 and ascale[rows][HN/128] f32 -- the layout rocwmma::fragment's row_major load
 * wants directly (ldm = HN), matching hc_down_pf2_fp8_kernel below. Grid: (rows, HN/128); block: 32 (one wave).
 *
 * Quantisation math validated against real ground-truth N: tools/lab/radiance/hc-pf2-quant-a8.hip (0
 * mismatches, 80 scales + 10240 codes; RMS error 2.53%). The N-building half transcribed here (bf2f * scale
 * * (w + wadd)) has NOT yet been separately re-measured against a fresh hardware dump of real h/scalein/w --
 * it is proven correct by being the same formula stageA already runs, not by a new standalone harness; that
 * harness is the next thing to build if this kernel's output is doubted beyond code review. */
__global__ __launch_bounds__(32) void hc_pf2_quant_a8_kernel(
    const unsigned short* __restrict__ h, const float* __restrict__ scalein,
    const float* __restrict__ w, unsigned char* __restrict__ a8_out,
    float* __restrict__ ascale_out, long n, int hc, long hrow, float wadd) {
    const long row = blockIdx.x, g = blockIdx.y;
    const int lane = threadIdx.x;
    const long HN = (long)hc * n;
    const long k0 = g * 128;
    const int c = (int)(k0 / n);
    const float sc = scalein[row * kHcMax + c];
    const unsigned short* hp = h + row * hrow + k0;
    const float* wp = w + k0;
    float v[4], local_max = 0.0f;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        v[e] = bf2f(hp[lane * 4 + e]) * sc * (wp[lane * 4 + e] + wadd);
        local_max = fmaxf(local_max, fabsf(v[e]));
    }
    for (int off = 16; off; off >>= 1) local_max = fmaxf(local_max, __shfl_xor(local_max, off));
    const float scale = local_max > 0.0f ? local_max / 448.0f : 1.0f;
    if (lane == 0) ascale_out[row * (HN / 128) + g] = scale;
    unsigned char* out = a8_out + row * HN + k0;
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
 * (store_matrix_sync to a small LDS buffer) before the rescale-and-accumulate. Each wave's part_tile slot is
 * indexed by wave (8 waves run concurrently): an unindexed shared buffer here lets every wave overwrite the
 * others' partials -- this exact bug was caught and fixed twice this session in the standalone harnesses.
 *
 * Output contract copied verbatim from the real hc_down_pf2_kernel's epilogue (r4d_hc_bf16.hip): when ks > 1,
 * write the UNACTIVATED value into part[(blockIdx.y*rows+r)*ncol+col] for hc_down_reduce_kernel to sum (hence
 * `ncol`, not `lr`, is this kernel's part[] stride, matching that reduce kernel even though this kernel only
 * ever writes columns < lr itself); else write dout[r*lr+col] = f2bf(siluf_(v*inv_hc)) directly. inv_hc and
 * wadd: note wadd is NOT applied here -- in the real kernel it is folded into N while building it (stageA's
 * `(pw[e] + wadd)`), not in the down-product epilogue; this kernel inherits wadd in its signature only to stay
 * a drop-in alternative at the call site, unused until the N-side of this pipeline needs it (see below).
 *
 * Validated end to end at REAL production scale (RT=64, lr=320/20 column tiles, kWaves=8, real activation and
 * real W_down bytes): tools/lab/radiance/hc-down-fp8-fullscale.hip, max rel diff 2.7e-5 (fp32 noise floor)
 * against a quantised-operand reference, for the ks==1 (kc >= HN, no split-K) direct-to-dout path exercised
 * there. The ks>1/part[] path below uses the real kernel's own indexing convention (confirmed by reading
 * hc_down_pf2_kernel and hc_down_reduce_kernel directly) but has NOT been separately exercised by a harness
 * that drives hc_down_reduce_kernel afterward -- treat it as transcribed-and-reviewed, not measured, until
 * that harness exists.
 *
 * KNOWN GAP, found reading the real source for this revision (not yet addressed by any kernel in this patch):
 * hc_pf2_quant_a8_kernel above assumes a materialized bf16 N array already exists to quantise. No such array
 * exists anywhere in the real pipeline -- hc_down_pf2_kernel builds N incrementally, tile by tile, straight
 * from `h` + per-stream norm scale + gamma weight `w` (`fetchA`/`stageA`), into a small double-buffered LDS
 * window, and never materializes the whole thing. Before hc_pf2_quant_a8_kernel can run for real, a separate
 * kernel (or an extension of hc_pf2_prep_kernel) must build and store the FULL bf16 N array first. That
 * kernel does not exist yet; this is the next real blocker, not the dispatch wiring. */
__global__ __launch_bounds__(kThreads) void hc_down_pf2_fp8_kernel(
    const unsigned char* __restrict__ a8, const float* __restrict__ ascale,
    const unsigned char* __restrict__ down, long dld,
    unsigned short* __restrict__ dout, float* __restrict__ part,
    long rows, long lr, long ncol, long kc, long HN, float wadd) {
    using namespace rocwmma;
    using Afrag = fragment<matrix_a, 16, 16, 16, float8_t, row_major>;
    using Bfrag = fragment<matrix_b, 16, 16, 16, float8_t, col_major>;
    using Cfrag = fragment<accumulator, 16, 16, 16, float>;

    constexpr int RT = kPf2RT, MT = RT / 16;
    const int wave = threadIdx.x / kWave, lane = threadIdx.x % kWave;
    const long ntt = (lr + 15) / 16;
    const long r0 = blockIdx.x * RT;
    const long ka = blockIdx.y * kc;
    const long kb = (ka + kc < HN) ? ka + kc : HN;
    if (r0 >= rows || ka >= HN) return;
    const long nkb0 = ka / 128, nkb1 = (kb + 127) / 128;
    const long nkbTotal = (HN + 127) / 128;
    const float inv_hc = 1.0f / 4.0f;  /* hc is fixed at 4 for this pinned shape, matching the real kernel's call site */
    (void)wadd;

    __shared__ float part_tile[kWaves][16][16];
    for (long nt = wave; nt < ntt; nt += kWaves) {
        const long col0 = nt * 16;
        for (int m = 0; m < MT; ++m) {
            const long row0 = r0 + m * 16;
            if (row0 >= rows) break;
            float total[8] = {};
            for (long g = nkb0; g < nkb1; ++g) {
                Cfrag fc;
                fill_fragment(fc, 0.0f);
                for (int s = 0; s < 128 / 16; ++s) {
                    const long k0 = g * 128 + s * 16;
                    Afrag fa; Bfrag fb;
                    load_matrix_sync(fa, reinterpret_cast<const float8_t*>(a8 + (size_t)row0 * HN + k0), HN);
                    load_matrix_sync(fb, reinterpret_cast<const float8_t*>(down + (size_t)col0 * dld + k0), dld);
                    mma_sync(fc, fa, fb, fc);
                }
                store_matrix_sync(&part_tile[wave][0][0], fc, 16, mem_row_major);
                __syncwarp();
#pragma unroll
                for (int t = 0; t < 8; ++t) {
                    const int idx = lane + t * kWave;
                    const int i = idx / 16, j = idx % 16;
                    const long row = row0 + i, col = col0 + j;
                    if (row >= rows || col >= lr) continue;
                    const float asc = ascale[(size_t)row * nkbTotal + g];
                    const float* wsc = reinterpret_cast<const float*>(down + (size_t)col * dld + HN);
                    total[t] += part_tile[wave][i][j] * asc * wsc[g];
                }
                __syncwarp();
            }
#pragma unroll
            for (int t = 0; t < 8; ++t) {
                const int idx = lane + t * kWave;
                const int i = idx / 16, j = idx % 16;
                const long row = row0 + i, col = col0 + j;
                if (row >= rows || col >= lr) continue;
                const float v = total[t];
                if (part) part[((long)blockIdx.y * rows + row) * ncol + col] = v;
                else      dout[row * lr + col] = f2bf(siluf_(v * inv_hc));
            }
        }
    }
}
'''

_INJ_KERNEL = '''
/* Injection gains [0, hc): hc is 4 for this pinned shape, far too narrow to tile with WMMA (a 16x16 tile
 * would be 75% padding), so this is a plain per-(row, column) wave reduction instead of hc_down_pf2_kernel's
 * WMMA shape -- same formula (2*sigmoid(dot(N, injw_c) * inv_hc)), same inputs, just not the same tiling,
 * since there is no FP8 variant of injw to tile against. Reads N back from the SAME quantised a8/ascale this
 * patch's down-product reads -- deliberately: both consumers must see the identical (lossy) activation this
 * patch introduces, or down and inject would drift against two different approximations of N. One wave (32
 * lanes) per (row, c): each lane strides HN in 32s, dequantising a8*ascale and multiplying by injw (bf2f),
 * then a full-wave sum closes it out. hc=4 columns per row is a trivial fraction of the down-product's
 * HN*lr multiply-adds, so a serial per-lane loop (no WMMA, no tiling) is proportionate, not a performance
 * risk. wadd is NOT re-applied here: it is already baked into a8 by hc_pf2_quant_a8_kernel above.
 *
 * NOT yet validated against real ground-truth inject gains on hardware -- mechanically transcribed from the
 * real epilogue's formula and reviewed, same confidence tier as the N-build half of the quant kernel above. */
__global__ __launch_bounds__(32) void hc_inj_pf2_bf16_kernel(
    const unsigned char* __restrict__ a8, const float* __restrict__ ascale,
    const unsigned short* __restrict__ injw, unsigned short* __restrict__ inj,
    long rows, long HN, int hc, long irow) {
    const long row = blockIdx.x;
    const int c = (int)blockIdx.y, lane = threadIdx.x;
    if (row >= rows || c >= hc) return;
    const long nkb = HN / 128;
    const unsigned char* arow = a8 + (size_t)row * HN;
    const float* asc = ascale + (size_t)row * nkb;
    const unsigned short* wr = injw + (size_t)c * HN;
    float acc = 0.0f;
    for (long k = lane; k < HN; k += 32) {
        const float av = float(*reinterpret_cast<const rocwmma::float8_t*>(&arow[k])) * asc[k / 128];
        acc += av * bf2f(wr[k]);
    }
    for (int off = 16; off; off >>= 1) acc += __shfl_xor(acc, off);
    if (lane == 0) inj[row * irow + c] = f2bf(2.0f * sigmoidf_(acc * (1.0f / (float)hc)));
}
'''

_REDUCE_KERNEL = '''
/* RAD17: the FP8 down-product's own split-K reduce. hc_down_reduce_kernel (the real one) sums BOTH the
 * down and inject columns of `part[]` in one pass, but this pipeline never writes inject columns into
 * `part[]` at all -- hc_inj_pf2_bf16_kernel above computes inj directly, in one shot, independent of K-split
 * (its cost is trivial; splitting it would be pure overhead). Reusing hc_down_reduce_kernel unmodified would
 * read uninitialised `part[]` for every inject column and corrupt `inj`, so this is a separate, down-only
 * reduce scoped to [rows x lr], not [rows x ncol]. Mechanically the same accumulation hc_down_reduce_kernel
 * already runs for its own columns (ascending z, matching add order); not yet measured on real split-K
 * shapes -- every PF2-eligible shape this session found has ks > 1 (M=512 was already ks=7), so this is the
 * piece that actually matters for any real call, not an edge case. */
__global__ __launch_bounds__(kThreads) void hc_down_pf2_fp8_reduce_kernel(
    const float* __restrict__ part, unsigned short* __restrict__ dout,
    long rows, long lr, long ncol, int ksplit) {
    const long i = (long)blockIdx.x * kThreads + threadIdx.x;
    const long tot = rows * lr;
    if (i >= tot) return;
    const long row = i / lr, col = i - row * lr;
    const long base = row * ncol + col, stride = rows * ncol;
    float s = 0.0f;
    for (int z = 0; z < ksplit; ++z) s += part[(long)z * stride + base];
    dout[row * lr + col] = f2bf(siluf_(s * 0.25f));
}
'''

_DISPATCH_HELPERS = '''
/* RAD17: dispatch for the FP8 PF2 down-product above, default off (RAD_HC_PF2_FP8_DOWN=1 to enable).
 * Kept on its own lazily-grown scratch allocation, separate from r4d_rad_hc_read_scratch's budgeted `u`/`wsp`
 * accounting, until this path has real hardware acceptance evidence -- growing a brand-new allocation here
 * cannot corrupt the existing, already-correct scratch sizing for the bf16 path. Grown, never shrunk; freed
 * only at process exit, matching this file's other static device buffers' lifetime. */
static bool g_hc_pf2_fp8_down = [] {
    const char* v = std::getenv("RAD_HC_PF2_FP8_DOWN");
    return v && v[0] == '1' && v[1] == '\\0';
}();
static unsigned char* g_fp8a_scratch = nullptr;
static size_t         g_fp8a_scratch_bytes = 0;
static bool hc_pf2_fp8a_scratch(size_t need, unsigned char** out) {
    if (need > g_fp8a_scratch_bytes) {
        if (g_fp8a_scratch) (void)hipFree(g_fp8a_scratch);
        g_fp8a_scratch = nullptr;
        g_fp8a_scratch_bytes = 0;
        if (hipMalloc(&g_fp8a_scratch, need) != hipSuccess) return false;
        g_fp8a_scratch_bytes = need;
    }
    *out = g_fp8a_scratch;
    return true;
}
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
            text=_QUANT_A8_KERNEL + _FP8_DOWN_KERNEL + _INJ_KERNEL + _REDUCE_KERNEL + _DISPATCH_HELPERS
                 + "\nstatic int hc_pf2_launch(",
            guard=r"hc_pf2_quant_a8_kernel",
            expect_matches=1,
            rationale="Unique declaration of the PF2 launcher; the new kernels and dispatch helpers must "
                       "exist before it for the call-site edit below to switch between paths",
        ),
        Edit(
            id="hc-pf2-fp8-down-dispatch",
            anchor=(
                r"    float\* const kpart = ks > 1 \? part : nullptr;\n"
                r"    hipLaunchKernelGGL\(hc_down_pf2_kernel,\n"
                r"                       dim3\(\(unsigned\)\(\(rows \+ kPf2RT - 1\) / kPf2RT\), \(unsigned\)ks\),\n"
                r"                       dim3\(kThreads\), 0, \(hipStream_t\)s, h, w, \(const uint4\*\)plane,\n"
                r"                       \(const float\*\)scale, dbuf, inj, kpart, rows, n, hc, lr, ncol, ntt, hrow,\n"
                r"                       irow, kc, wadd\);\n"
                r"    if \(hipGetLastError\(\) != hipSuccess\) return RAD_E_DEVICE;\n"
            ),
            mode="replace",
            text=(
                "    float* const kpart = ks > 1 ? part : nullptr;\n"
                "    /* RAD17: every real PF2-eligible shape found this session has ks > 1 (M=512 already has\n"
                "     * ks=7), so an eligibility check that required ks==1 would make this path dead code for\n"
                "     * every real call -- it is not restricted to ks==1. The allocator-failure case still falls\n"
                "     * through to the original bf16 call below unchanged. */\n"
                "    const bool fp8_down_eligible = g_hc_pf2_fp8_down && f8;\n"
                "    unsigned char* fp8a = nullptr;\n"
                "    bool fp8_down_ok = false;\n"
                "    if (fp8_down_eligible) {\n"
                "        const long nkb = (HN + 127) / 128;\n"
                "        const size_t a8_bytes = (size_t)rows * HN;\n"
                "        const size_t aligned = (a8_bytes + 15) & ~(size_t)15;\n"
                "        const size_t need = aligned + (size_t)rows * nkb * sizeof(float);\n"
                "        if (hc_pf2_fp8a_scratch(need, &fp8a)) {\n"
                "            float* const ascale = (float*)(fp8a + aligned);\n"
                "            hipLaunchKernelGGL(hc_pf2_quant_a8_kernel, dim3((unsigned)rows, (unsigned)nkb),\n"
                "                               dim3(32), 0, (hipStream_t)s, h, scale, w, fp8a, ascale, n, hc,\n"
                "                               hrow, wadd);\n"
                "            if (hipGetLastError() != hipSuccess) return RAD_E_DEVICE;\n"
                "            hipLaunchKernelGGL(hc_down_pf2_fp8_kernel,\n"
                "                               dim3((unsigned)((rows + kPf2RT - 1) / kPf2RT), (unsigned)ks),\n"
                "                               dim3(kThreads), 0, (hipStream_t)s, fp8a, ascale,\n"
                "                               (const unsigned char*)down, dld, dbuf, kpart, rows, lr, ncol,\n"
                "                               kc, HN, wadd);\n"
                "            if (hipGetLastError() != hipSuccess) return RAD_E_DEVICE;\n"
                "            if (ks > 1) {\n"
                "                const long totlr = rows * lr;\n"
                "                hipLaunchKernelGGL(hc_down_pf2_fp8_reduce_kernel,\n"
                "                                   dim3((unsigned)((totlr + kThreads - 1) / kThreads)),\n"
                "                                   dim3(kThreads), 0, (hipStream_t)s, (const float*)kpart,\n"
                "                                   dbuf, rows, lr, ncol, ks);\n"
                "                if (hipGetLastError() != hipSuccess) return RAD_E_DEVICE;\n"
                "            }\n"
                "            if (injw && inj) {\n"
                "                hipLaunchKernelGGL(hc_inj_pf2_bf16_kernel, dim3((unsigned)rows, (unsigned)hc),\n"
                "                                   dim3(32), 0, (hipStream_t)s, fp8a, ascale, injw, inj, rows,\n"
                "                                   HN, hc, irow);\n"
                "                if (hipGetLastError() != hipSuccess) return RAD_E_DEVICE;\n"
                "            }\n"
                "            fp8_down_ok = true;\n"
                "        }\n"
                "    }\n"
                "    if (!fp8_down_ok) {\n"
                "    hipLaunchKernelGGL(hc_down_pf2_kernel,\n"
                "                       dim3((unsigned)((rows + kPf2RT - 1) / kPf2RT), (unsigned)ks),\n"
                "                       dim3(kThreads), 0, (hipStream_t)s, h, w, (const uint4*)plane,\n"
                "                       (const float*)scale, dbuf, inj, kpart, rows, n, hc, lr, ncol, ntt, hrow,\n"
                "                       irow, kc, wadd);\n"
                "    if (hipGetLastError() != hipSuccess) return RAD_E_DEVICE;\n"
                "    }\n"
            ),
            guard=r"fp8_down_eligible",
            expect_matches=1,
            rationale="Unique down-product launch call site inside hc_pf2_launch; this is the only place "
                       "that decides which down-product kernel actually runs",
        ),
    ),
)
