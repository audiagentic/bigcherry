# VTP37: gfx1100 K-parallel look-ahead qualification

## Stage 1 — compile-time occupancy rule and resource gate: PASS

Only the gfx1100 native rewrite's K-parallel body is changed. The formula
counts the simultaneously live register arrays and the weight/token
look-ahead rings from their actual C++ fragment types (Stage8, i2v, f8v,
r3_i8), using 32-bit words. Predicted wave residency uses the gfx1100
wave32 register file (1,536 VGPR per SIMD). The predictor is deliberately
restricted to NT <= SB/128: with extra token tiles, the compiler's
overlapping inter-tile bq loads cause the resource records (SB128 NT2,
NT3/4) to deviate from this simple model. For all KW1/2/4 and both
chain modes, the consequent changed set is SB128 NT1 and SB256 NT1/2,
18 of the 48 instantiations. This is a constexpr rule over the body types
and SB/NT, not a lookup by model or N/K/M. All other instantiations compile
the stock look-ahead.

The previous acc-once WMMA path, scale order and SB256 multi-tile unroll
are untouched. The old R4D_WAVES(6) hint is retained: actual compiler
allocation for other shapes is not predictable from these live type sizes
well enough to derive a safe new waves-per-EU hint. The v7-style
gain from a fixed higher hint is therefore **not shipped**.

An isolated gfx1100 libr3 unit build completed (nice -n 19, -j1).
Extracted AMDGPU metadata from the built plugin and the 2020.8.3
baseline. Every kpar instantiation has zero new spills, zero increased
VGPR/private scratch and zero occupancy losses. All 278 non-kpar kernels
in the same unit retain identical resource metadata. Resource table
cells are VGPR/spill/private-bytes/theoretical waves per SIMD.

| KW | SB | NT | chain | 2020.8.3 VGPR/spill/private/waves | candidate VGPR/spill/private/waves |
|---:|---:|---:|---|---|---|
| 1 | 128 | 1 | ch | 143/0/0/10 | 117/0/0/13 |
| 1 | 128 | 1 | un | 140/0/0/10 | 114/0/0/13 |
| 1 | 128 | 2 | ch | 177/0/0/8 | 177/0/0/8 |
| 1 | 128 | 2 | un | 174/0/0/8 | 174/0/0/8 |
| 1 | 128 | 3 | ch | 240/2/12/6 | 240/2/12/6 |
| 1 | 128 | 3 | un | 240/0/0/6 | 240/0/0/6 |
| 1 | 128 | 4 | ch | 179/0/0/8 | 179/0/0/8 |
| 1 | 128 | 4 | un | 176/0/0/8 | 176/0/0/8 |
| 1 | 256 | 1 | ch | 212/0/0/7 | 181/0/0/8 |
| 1 | 256 | 1 | un | 209/0/0/7 | 178/0/0/8 |
| 1 | 256 | 2 | ch | 176/0/0/8 | 154/0/0/9 |
| 1 | 256 | 2 | un | 173/0/0/8 | 151/0/0/10 |
| 1 | 256 | 3 | ch | 213/0/0/7 | 213/0/0/7 |
| 1 | 256 | 3 | un | 210/0/0/7 | 210/0/0/7 |
| 1 | 256 | 4 | ch | 173/0/0/8 | 173/0/0/8 |
| 1 | 256 | 4 | un | 170/0/0/9 | 170/0/0/9 |
| 2 | 128 | 1 | ch | 144/0/0/10 | 118/0/0/13 |
| 2 | 128 | 1 | un | 141/0/0/10 | 115/0/0/13 |
| 2 | 128 | 2 | ch | 178/0/0/8 | 178/0/0/8 |
| 2 | 128 | 2 | un | 175/0/0/8 | 175/0/0/8 |
| 2 | 128 | 3 | ch | 240/3/16/6 | 240/3/16/6 |
| 2 | 128 | 3 | un | 240/1/8/6 | 240/1/8/6 |
| 2 | 128 | 4 | ch | 180/0/0/8 | 180/0/0/8 |
| 2 | 128 | 4 | un | 177/0/0/8 | 177/0/0/8 |
| 2 | 256 | 1 | ch | 213/0/0/7 | 182/0/0/8 |
| 2 | 256 | 1 | un | 210/0/0/7 | 179/0/0/8 |
| 2 | 256 | 2 | ch | 177/0/0/8 | 155/0/0/9 |
| 2 | 256 | 2 | un | 174/0/0/8 | 152/0/0/10 |
| 2 | 256 | 3 | ch | 215/0/0/7 | 215/0/0/7 |
| 2 | 256 | 3 | un | 212/0/0/7 | 212/0/0/7 |
| 2 | 256 | 4 | ch | 174/0/0/8 | 174/0/0/8 |
| 2 | 256 | 4 | un | 171/0/0/8 | 171/0/0/8 |
| 4 | 128 | 1 | ch | 144/0/0/10 | 118/0/0/13 |
| 4 | 128 | 1 | un | 141/0/0/10 | 115/0/0/13 |
| 4 | 128 | 2 | ch | 178/0/0/8 | 178/0/0/8 |
| 4 | 128 | 2 | un | 175/0/0/8 | 175/0/0/8 |
| 4 | 128 | 3 | ch | 240/3/16/6 | 240/3/16/6 |
| 4 | 128 | 3 | un | 240/1/8/6 | 240/1/8/6 |
| 4 | 128 | 4 | ch | 180/0/0/8 | 180/0/0/8 |
| 4 | 128 | 4 | un | 177/0/0/8 | 177/0/0/8 |
| 4 | 256 | 1 | ch | 213/0/0/7 | 182/0/0/8 |
| 4 | 256 | 1 | un | 210/0/0/7 | 179/0/0/8 |
| 4 | 256 | 2 | ch | 177/0/0/8 | 155/0/0/9 |
| 4 | 256 | 2 | un | 174/0/0/8 | 152/0/0/10 |
| 4 | 256 | 3 | ch | 215/0/0/7 | 215/0/0/7 |
| 4 | 256 | 3 | un | 212/0/0/7 | 212/0/0/7 |
| 4 | 256 | 4 | ch | 174/0/0/8 | 174/0/0/8 |
| 4 | 256 | 4 | un | 171/0/0/8 | 171/0/0/8 |

## Stage 2 — bit-identical BF16 output: PASS

Same-process 2020.8.3 baseline and new single-unit plugin comparisons,
identical deterministic input/scales, complete output BF16 bytes over
every M 1–64 for the six TP3 per-rank shapes, ragged N=5136 and
short-K KW2 cases K=512 and K=256. Each test acquired and released
its own queued GPU lease (12-minute TTL, 5-minute estimate).

| Row/dispatch | PASS | SKIP | FAIL |
|---|---:|---:|---:|
| iq/unchained | 576 | 0 | 0 |
| iq/chained | 576 | 0 | 0 |
| i8/unchained | 576 | 0 | 0 |
| i8/chained | 576 | 0 | 0 |

**Total 2,304 PASS, 0 SKIP, 0 FAIL.**
No numerical divergence. No serving, ms/step A/B or KLD.

## Stage 3 — complete libr3 build and targeted r3_selftest: PASS

Full linked plugin, R3_NATIVE=ON, R3_ONLY unset, all 115 gfx1100 device units;
nice -n 19 cmake --build <build-full> --target libr3 r3_selftest -j 3
finished successfully, with zero stubbed libr4d symbols. The five
specified narrow/IQR selftest processes were actually run independently,
each under its own short queued GPU lease. Zero FAIL and zero SKIP.

| r3_selftest --case | PASS | FAIL | SKIP |
|---|---:|---:|---:|
| gemm_i8a8_nt_m16 | 13 | 0 | 0 |
| gemm_iq8r_nt_m16 | 35 | 0 | 0 |
| gemm_iqr_nt_m64 | 1034 | 0 | 0 |
| gemm_iqr_tiled | 129 | 0 | 0 |
| gemm_iqr_gated | 36 | 0 | 0 |

Total: 1247 PASS, 0 FAIL, 0 SKIP.

## Stage 4 — cold replay on one gfx1100 card

Baseline 2020.8.3 and full linked candidate, same driver, 256 MiB
cold-cache flush between timed launches, 30 launches per sample.
Three repeats per shape/arm; six shapes × eight M values × two arms
× three repeats = 288 JSONL records, each with iq8r and i8a8.
Each of the 12 shape/arm runs had its own queued GPU lease.

Resource-selected NT1 or NT2 rows were gated on no regression
outside combined min-max repeat spread; M20-32 on the four primary
shapes must be faster beyond that spread. NT3/NT4 are observations.

Gate/up IQ8R M4 effective bytes 69222400; before 771.9 GB/s, after 782.3 GB/s (int8 weights plus BF16 row scales).
Gate/up IQ8R M24 effective bytes 69222400; before 711.3 GB/s, after 748.2 GB/s (int8 weights plus BF16 row scales).

Median regressions beyond repeat spread: 2
Primary-band improvements failing spread gate: 15
NT1 rows not faster in median: 1

NOT READY — performance gate not met

### Per-shape timing matrix

| N x K | M | row | 2020.8.3 us median [min-max] | new us median [min-max] | change |
|---|---:|---|---|---|---:|
| 13312x5120 | 4 | iq8r | 89.68 [89.56-89.72] | 88.48 [88.32-88.52] | -1.34% |
| 13312x5120 | 4 | i8a8 | 90.48 [90.20-92.52] | 88.92 [88.68-90.28] | -1.72% |
| 13312x5120 | 16 | iq8r | 91.32 [91.16-91.32] | 90.56 [90.52-90.60] | -0.83% |
| 13312x5120 | 16 | i8a8 | 92.40 [92.24-92.44] | 90.92 [90.76-91.00] | -1.60% |
| 13312x5120 | 20 | iq8r | 96.68 [96.64-97.24] | 92.00 [91.80-92.16] | -4.84% |
| 13312x5120 | 20 | i8a8 | 98.64 [98.40-98.80] | 98.52 [98.32-99.00] | -0.12% |
| 13312x5120 | 24 | iq8r | 97.32 [97.28-97.52] | 92.52 [92.52-92.52] | -4.93% |
| 13312x5120 | 24 | i8a8 | 99.32 [98.96-99.56] | 99.40 [98.96-99.52] | +0.08% |
| 13312x5120 | 28 | iq8r | 98.32 [98.12-98.52] | 93.16 [93.04-93.32] | -5.25% |
| 13312x5120 | 28 | i8a8 | 100.00 [99.92-100.12] | 100.24 [99.52-100.36] | +0.24% |
| 13312x5120 | 32 | iq8r | 98.92 [98.84-99.24] | 93.60 [93.48-93.88] | -5.38% |
| 13312x5120 | 32 | i8a8 | 100.44 [100.40-100.56] | 100.88 [100.88-100.92] | +0.44% |
| 13312x5120 | 40 | iq8r | 113.68 [113.28-113.92] | 113.36 [113.08-113.84] | -0.28% |
| 13312x5120 | 40 | i8a8 | 120.40 [119.92-120.76] | 120.36 [120.36-120.72] | -0.03% |
| 13312x5120 | 48 | iq8r | 114.20 [114.16-114.40] | 114.12 [114.12-114.12] | -0.07% |
| 13312x5120 | 48 | i8a8 | 121.24 [120.92-121.40] | 121.56 [121.08-121.56] | +0.26% |
| 5120x6656 | 4 | iq8r | 51.04 [50.60-51.16] | 47.16 [47.12-47.20] | -7.60% |
| 5120x6656 | 4 | i8a8 | 50.00 [49.92-53.08] | 49.04 [49.00-51.88] | -1.92% |
| 5120x6656 | 16 | iq8r | 50.96 [50.88-51.28] | 47.48 [47.36-47.68] | -6.83% |
| 5120x6656 | 16 | i8a8 | 50.08 [50.08-50.28] | 50.60 [50.32-50.68] | +1.04% |
| 5120x6656 | 20 | iq8r | 57.24 [56.24-57.44] | 53.96 [53.80-54.12] | -5.73% |
| 5120x6656 | 20 | i8a8 | 53.84 [53.68-54.00] | 53.48 [53.32-53.68] | -0.67% |
| 5120x6656 | 24 | iq8r | 58.12 [58.08-58.12] | 53.84 [53.76-53.92] | -7.36% |
| 5120x6656 | 24 | i8a8 | 53.40 [53.32-53.96] | 53.48 [53.28-53.88] | +0.15% |
| 5120x6656 | 28 | iq8r | 57.80 [57.56-58.20] | 54.04 [54.00-54.28] | -6.51% |
| 5120x6656 | 28 | i8a8 | 54.92 [54.72-55.12] | 53.72 [53.68-53.88] | -2.18% |
| 5120x6656 | 32 | iq8r | 58.28 [57.92-58.36] | 53.92 [53.84-54.00] | -7.48% |
| 5120x6656 | 32 | i8a8 | 55.16 [54.68-55.32] | 54.96 [54.84-55.92] | -0.36% |
| 5120x6656 | 40 | iq8r | 65.12 [64.84-65.16] | 64.96 [64.84-65.32] | -0.25% |
| 5120x6656 | 40 | i8a8 | 66.00 [65.76-66.00] | 65.88 [65.56-65.92] | -0.18% |
| 5120x6656 | 48 | iq8r | 65.36 [65.12-65.44] | 65.36 [65.08-65.44] | +0.00% |
| 5120x6656 | 48 | i8a8 | 65.92 [65.56-66.12] | 65.76 [65.68-65.88] | -0.24% |
| 6144x5120 | 4 | iq8r | 45.04 [44.88-45.36] | 43.48 [43.44-43.56] | -3.46% |
| 6144x5120 | 4 | i8a8 | 44.16 [44.00-47.44] | 43.56 [43.44-46.44] | -1.36% |
| 6144x5120 | 16 | iq8r | 45.28 [45.24-45.48] | 44.16 [44.12-44.40] | -2.47% |
| 6144x5120 | 16 | i8a8 | 45.12 [44.92-45.20] | 44.52 [44.32-44.64] | -1.33% |
| 6144x5120 | 20 | iq8r | 49.76 [49.76-50.00] | 46.88 [46.60-46.92] | -5.79% |
| 6144x5120 | 20 | i8a8 | 48.24 [48.16-48.24] | 48.12 [47.92-48.36] | -0.25% |
| 6144x5120 | 24 | iq8r | 50.12 [50.00-50.32] | 46.56 [46.56-46.92] | -7.10% |
| 6144x5120 | 24 | i8a8 | 48.28 [48.20-48.36] | 48.32 [48.32-48.36] | +0.08% |
| 6144x5120 | 28 | iq8r | 50.36 [50.32-50.56] | 46.36 [46.28-46.80] | -7.94% |
| 6144x5120 | 28 | i8a8 | 48.48 [48.44-48.48] | 48.32 [48.20-48.36] | -0.33% |
| 6144x5120 | 32 | iq8r | 50.56 [50.40-50.76] | 46.68 [46.52-46.76] | -7.67% |
| 6144x5120 | 32 | i8a8 | 48.60 [48.44-48.88] | 48.44 [48.20-48.60] | -0.33% |
| 6144x5120 | 40 | iq8r | 57.52 [57.40-57.64] | 57.36 [57.24-57.60] | -0.28% |
| 6144x5120 | 40 | i8a8 | 60.36 [60.32-60.44] | 60.44 [60.36-60.48] | +0.13% |
| 6144x5120 | 48 | iq8r | 57.60 [57.56-57.64] | 57.64 [57.28-57.80] | +0.07% |
| 6144x5120 | 48 | i8a8 | 60.84 [60.76-61.00] | 60.68 [60.64-60.80] | -0.26% |
| 5120x2304 | 4 | iq8r | 24.52 [24.44-24.56] | 22.32 [22.32-22.60] | -8.97% |
| 5120x2304 | 4 | i8a8 | 23.24 [23.20-25.96] | 22.56 [22.40-24.76] | -2.93% |
| 5120x2304 | 16 | iq8r | 24.40 [24.40-24.44] | 22.40 [22.40-22.56] | -8.20% |
| 5120x2304 | 16 | i8a8 | 23.40 [23.28-23.48] | 22.56 [22.56-22.56] | -3.59% |
| 5120x2304 | 20 | iq8r | 30.60 [30.52-30.76] | 26.56 [26.44-26.64] | -13.20% |
| 5120x2304 | 20 | i8a8 | 28.20 [28.16-28.20] | 28.36 [28.36-28.48] | +0.57% |
| 5120x2304 | 24 | iq8r | 30.68 [30.48-30.68] | 26.52 [26.48-26.56] | -13.56% |
| 5120x2304 | 24 | i8a8 | 28.32 [28.28-28.48] | 28.48 [28.24-28.56] | +0.56% |
| 5120x2304 | 28 | iq8r | 30.80 [30.68-30.96] | 26.56 [26.44-26.60] | -13.77% |
| 5120x2304 | 28 | i8a8 | 28.08 [27.96-28.40] | 28.32 [28.20-28.36] | +0.86% |
| 5120x2304 | 32 | iq8r | 30.60 [30.40-30.72] | 26.48 [26.40-26.52] | -13.46% |
| 5120x2304 | 32 | i8a8 | 28.00 [27.92-28.04] | 28.00 [28.00-28.04] | +0.00% |
| 5120x2304 | 40 | iq8r | 34.64 [34.56-34.68] | 34.56 [34.56-34.80] | -0.23% |
| 5120x2304 | 40 | i8a8 | 32.92 [32.88-33.04] | 33.00 [32.92-33.12] | +0.25% |
| 5120x2304 | 48 | iq8r | 34.72 [34.64-34.84] | 34.88 [34.88-34.96] | +0.46% |
| 5120x2304 | 48 | i8a8 | 33.20 [33.08-33.36] | 33.20 [33.16-33.32] | +0.00% |
| 3072x5120 | 4 | iq8r | 26.44 [26.24-26.60] | 26.00 [25.96-26.08] | -1.67% |
| 3072x5120 | 4 | i8a8 | 26.48 [26.36-28.36] | 25.76 [25.64-27.24] | -2.72% |
| 3072x5120 | 16 | iq8r | 26.48 [26.44-26.56] | 26.24 [26.16-26.32] | -0.90% |
| 3072x5120 | 16 | i8a8 | 26.48 [26.40-26.52] | 25.80 [25.72-25.92] | -2.57% |
| 3072x5120 | 20 | iq8r | 28.48 [28.36-28.48] | 27.08 [27.00-27.20] | -4.92% |
| 3072x5120 | 20 | i8a8 | 28.16 [28.04-28.20] | 28.04 [28.04-28.04] | -0.43% |
| 3072x5120 | 24 | iq8r | 28.52 [28.48-28.56] | 27.12 [27.08-27.20] | -4.91% |
| 3072x5120 | 24 | i8a8 | 28.12 [28.04-28.48] | 28.24 [28.12-28.28] | +0.43% |
| 3072x5120 | 28 | iq8r | 28.56 [28.48-28.80] | 27.24 [27.12-27.24] | -4.62% |
| 3072x5120 | 28 | i8a8 | 28.64 [28.48-28.64] | 28.52 [28.16-28.52] | -0.42% |
| 3072x5120 | 32 | iq8r | 28.84 [28.72-28.92] | 27.36 [27.32-27.40] | -5.13% |
| 3072x5120 | 32 | i8a8 | 28.76 [28.56-28.80] | 28.92 [28.52-28.96] | +0.55% |
| 3072x5120 | 40 | iq8r | 33.40 [33.16-33.44] | 33.44 [33.28-33.48] | +0.12% |
| 3072x5120 | 40 | i8a8 | 35.16 [35.12-35.36] | 35.12 [35.04-35.20] | -0.11% |
| 3072x5120 | 48 | iq8r | 33.60 [33.40-33.80] | 33.92 [33.56-33.92] | +0.95% |
| 3072x5120 | 48 | i8a8 | 35.20 [35.04-35.24] | 35.24 [35.16-35.24] | +0.11% |
| 5120x1536 | 4 | iq8r | 18.96 [18.96-19.08] | 17.68 [17.68-17.72] | -6.75% |
| 5120x1536 | 4 | i8a8 | 18.36 [18.32-20.56] | 17.24 [17.24-18.20] | -6.10% |
| 5120x1536 | 16 | iq8r | 19.08 [19.08-19.08] | 17.48 [17.32-17.64] | -8.39% |
| 5120x1536 | 16 | i8a8 | 18.24 [18.08-18.32] | 17.28 [17.16-17.32] | -5.27% |
| 5120x1536 | 20 | iq8r | 23.92 [23.76-24.00] | 20.08 [20.00-20.24] | -16.05% |
| 5120x1536 | 20 | i8a8 | 22.08 [21.56-22.12] | 22.04 [21.96-22.08] | -0.18% |
| 5120x1536 | 24 | iq8r | 23.92 [23.92-24.12] | 20.32 [20.28-20.36] | -15.05% |
| 5120x1536 | 24 | i8a8 | 22.20 [22.04-22.32] | 22.24 [22.24-22.28] | +0.18% |
| 5120x1536 | 28 | iq8r | 23.80 [23.76-23.88] | 20.08 [20.00-20.08] | -15.63% |
| 5120x1536 | 28 | i8a8 | 22.12 [22.12-22.24] | 22.16 [22.04-22.16] | +0.18% |
| 5120x1536 | 32 | iq8r | 23.96 [23.96-24.00] | 20.24 [20.20-20.24] | -15.53% |
| 5120x1536 | 32 | i8a8 | 22.16 [22.00-22.28] | 22.16 [22.16-22.24] | +0.00% |
| 5120x1536 | 40 | iq8r | 28.24 [28.24-28.24] | 28.28 [28.20-28.40] | +0.14% |
| 5120x1536 | 40 | i8a8 | 26.08 [26.08-26.08] | 26.16 [26.16-26.16] | +0.31% |
| 5120x1536 | 48 | iq8r | 28.32 [27.96-28.40] | 28.32 [28.28-28.36] | +0.00% |
| 5120x1536 | 48 | i8a8 | 26.12 [26.00-26.28] | 26.28 [26.16-26.36] | +0.61% |

## Stage 5 — other look-ahead opportunities and disposition (inventory only)

**Final disposition: NOT READY, performance acceptance failed.** Do not push,
merge or ship this variant. The failed gate is independently reproducible
from 288 replay JSONL records and the per-row, per-shape table above.

The change meaningfully improves the IQ8R row in the M20-32 band:
at M24, gate_up 97.32 -> 92.52 us (-4.93%), down 58.12 -> 53.84 us
(-7.36%), inz 50.12 -> 46.56 us (-7.10%), out 30.68 -> 26.52 us
(-13.56%). By contrast, the I8A8 NT2 path with SB128 intentionally
remains stock under the model's conservative selector. Its M24 change
is within noise or slightly worse (gate_up +0.08%, down +0.15%,
inz +0.08%, out +0.56%), so it cannot satisfy the per-row M17-32
improvement requirement. Two I8A8 comparisons exceed the repeat
spread and one NT1 sample did not improve. These are not bit
divergences: the BF16 2,304-case identity sweep passed.

The live-register model is not a reliable enough predictor to admit
SB128 NT2 simply to chase the performance gate: its 175->173 unchained
VGPR change leaves compiled occupancy at 8 waves/SIMD, while NT3/4
behaviour depends on compiler liveness. Enabling NT2 would require
a new justified constexpr model and another complete resource,
bit-identity, build, selftest and cold-replay qualification.
No per-shape exceptions or occupancy-hint literals were added here.

Read-only inventory of other register rings in this compilation unit:

* The narrow 8-bit/FP8 GEMM (r4d_gemm_fp8a8.hip) uses
  Stage8 pf[kAF], pfn[kAF] for a one-K-tile-ahead weight ring,
  with pb[kBP] activation staging and qws/qas scale look-ahead.
  Potential pressure but source explicitly documents the prefetch
  as load-bearing; independent evidence is needed. Not changed.
* The tiled prefill in r4d_gemm_8a8_prefill.h uses
  Stage8 w0[kPTW], w1[kPTW] as a next-tile weight ring. Source
  evidence says trying two tiles ahead spills at 240 VGPR
  and slowed 25-38%; removing this ring has not been measured.
  Not changed.
* Scale look-ahead, LDS reduction and shim-exchange variations
  were excluded: prior ablations found non-identical numerics.
  No changes were made to these paths or other sessions' files.

**What actually ran:** isolated gfx1100 compilation, full 115-unit
gfx1100 libr3 + r3_selftest build, five narrow/IQR selftest cases
(1,247 PASS, zero FAIL/SKIP), four independent bit-check processes
(2,304 PASS, zero FAIL/SKIP) and 12 separately leased cold-replay
shape/arm runs yielding 288 records. No serving, ms/step A/B, KLD,
push or ship was performed.

