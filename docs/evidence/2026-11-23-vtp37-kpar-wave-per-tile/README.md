# VTP37 — K-parallel wave-per-token-tile experiment

**Decision: STOP / FAIL. Not suitable for merge or shipping.** Registers and bit identity pass, but the required cold effective-bandwidth gate and gate-up latency target fail. This is a negative experimental branch for lead review, not a production improvement.

## Scope and implementation

The native gfx1100 libr3 rewrite of r4d_gemm_fp8a8.hip introduces a special K-parallel body at NT >= 2: wave w,j owns one K stripe w and token tile j, blockDim = KW × NT × wave32, with the same grid N/16. The original NT=1 body, KW selection, WMMA/group/fmaf order and ascending-w LDS reduction remain unchanged. SB256 retains its earlier rolled unit loop. Each token tile independently re-reads weight fragments; no LDS weight staging. No IQR/vendor source, gfx1201 code, narrow tile, or prefill tile was edited.

## Code-object resource table

Each resource cell: **VGPR / spilled VGPR / private segment bytes / waves per SIMD** (wave32, floor(1536/VGPR)). Derived from the gfx1100 AMDGPU code-object metadata using llvm-readelf -n after llvm-objdump --offloading. The requested llvm-readobj --notes executable was not allowed by the remote command policy; llvm-readelf displayed the same metadata fields.

| KW | SB | NT | Mode | 2020.8.1 | Candidate |
|---:|---:|---:|---|---|---|
| 2 | 128 | 1 | chained | 151 / 0 / 0 / 10 | 151 / 0 / 0 / 10 |
| 2 | 128 | 1 | unchained | 148 / 0 / 0 / 10 | 148 / 0 / 0 / 10 |
| 2 | 128 | 2 | chained | 224 / 0 / 0 / 6 | 151 / 0 / 0 / 10 |
| 2 | 128 | 2 | unchained | 221 / 0 / 0 / 6 | 148 / 0 / 0 / 10 |
| 2 | 128 | 3 | chained | 240 / 3 / 16 / 6 | 151 / 0 / 0 / 10 |
| 2 | 128 | 3 | unchained | 240 / 2 / 12 / 6 | 148 / 0 / 0 / 10 |
| 2 | 128 | 4 | chained | 240 / 43 / 120 / 6 | 151 / 0 / 0 / 10 |
| 2 | 128 | 4 | unchained | 240 / 35 / 96 / 6 | 148 / 0 / 0 / 10 |
| 2 | 256 | 1 | chained | 203 / 0 / 0 / 7 | 203 / 0 / 0 / 7 |
| 2 | 256 | 1 | unchained | 200 / 0 / 0 / 7 | 200 / 0 / 0 / 7 |
| 2 | 256 | 2 | chained | 181 / 0 / 0 / 8 | 154 / 0 / 0 / 9 |
| 2 | 256 | 2 | unchained | 178 / 0 / 0 / 8 | 151 / 0 / 0 / 10 |
| 2 | 256 | 3 | chained | 210 / 0 / 0 / 7 | 154 / 0 / 0 / 9 |
| 2 | 256 | 3 | unchained | 207 / 0 / 0 / 7 | 151 / 0 / 0 / 10 |
| 2 | 256 | 4 | chained | 234 / 0 / 0 / 6 | 154 / 0 / 0 / 9 |
| 2 | 256 | 4 | unchained | 231 / 0 / 0 / 6 | 151 / 0 / 0 / 10 |
| 4 | 128 | 1 | chained | 151 / 0 / 0 / 10 | 151 / 0 / 0 / 10 |
| 4 | 128 | 1 | unchained | 148 / 0 / 0 / 10 | 148 / 0 / 0 / 10 |
| 4 | 128 | 2 | chained | 224 / 0 / 0 / 6 | 151 / 0 / 0 / 10 |
| 4 | 128 | 2 | unchained | 221 / 0 / 0 / 6 | 148 / 0 / 0 / 10 |
| 4 | 128 | 3 | chained | 240 / 3 / 16 / 6 | 151 / 0 / 0 / 10 |
| 4 | 128 | 3 | unchained | 240 / 2 / 12 / 6 | 148 / 0 / 0 / 10 |
| 4 | 128 | 4 | chained | 240 / 43 / 120 / 6 | 151 / 0 / 0 / 10 |
| 4 | 128 | 4 | unchained | 240 / 35 / 96 / 6 | 148 / 0 / 0 / 10 |
| 4 | 256 | 1 | chained | 203 / 0 / 0 / 7 | 203 / 0 / 0 / 7 |
| 4 | 256 | 1 | unchained | 200 / 0 / 0 / 7 | 200 / 0 / 0 / 7 |
| 4 | 256 | 2 | chained | 181 / 0 / 0 / 8 | 154 / 0 / 0 / 9 |
| 4 | 256 | 2 | unchained | 178 / 0 / 0 / 8 | 151 / 0 / 0 / 10 |
| 4 | 256 | 3 | chained | 210 / 0 / 0 / 7 | 154 / 0 / 0 / 9 |
| 4 | 256 | 3 | unchained | 207 / 0 / 0 / 7 | 151 / 0 / 0 / 10 |
| 4 | 256 | 4 | chained | 234 / 0 / 0 / 6 | 154 / 0 / 0 / 9 |
| 4 | 256 | 4 | unchained | 231 / 0 / 0 / 6 | 151 / 0 / 0 / 10 |

Resource gate: **PASS**. All KW=2 and KW=4 NT1 resource records are unchanged; NT2 reaches NT1's VGPR count or lower with zero spills; SB128 NT1–2 and SB256 NT1–4 have zero private scratch bytes. NT3–4 SB128 spills are removed. Maximum workgroup sizes are unchanged at NT1, and NT2/3/4 now use KW × 2/3/4 waves. KW1 was also checked as part of the 48 total instantiations. A separate metadata comparison found **278 of 278 non-K-parallel GEMM-unit kernels unchanged**, out of 326 total code-object kernel records, with none added or missing.

## Exact BF16 output-bit identity

**PASS: 2,304/2,304, zero skips or failures.** IQ8R SB256 and I8A8 SB128, each in chained and unchained dispatch, M 1–64 inclusive, all six drafter shapes 13312×5120, 5120×6656, 6144×5120, 5120×2304, 3072×5120 and 5120×1536; additionally ragged N 5136×5120 and short-K 5120×512, 5120×256 exercising KW2 as applicable. Each of the four arms: 576 passes. Baseline and candidate plugins loaded in one process with identical deterministic quantized inputs/scales, compared over full BF16 output bytes. The chained launch parameter was covered; no engine chain-signal run was attempted.

## Cold replay timing

One gfx1100 card; independent queued lease per arm, 256 MiB D2D cache flush before each timed launch, rotated weight and scale copies, 30 launches per measurement. Six shapes × eight M values × two arms × three repeats = **288 paired IQ8R/I8A8 records**, all collected. Medians [min–max] of three repetitions in microseconds; positive percent means regression. This is a standalone kernel ABI replay, not serving.

| N×K | M | Row | 2020.8.1 µs | Candidate µs | Delta |
|---|---:|---|---:|---:|---:|
| 13312x5120 | 4 | iq8r | 91.3 [91.3–91.4] | 91.2 [91.2–91.3] | -0.1% |
| 13312x5120 | 4 | i8a8 | 91.9 [91.7–96.4] | 91.6 [91.6–97.7] | -0.3% |
| 13312x5120 | 16 | iq8r | 93.5 [93.4–93.5] | 93.4 [93.3–93.7] | -0.1% |
| 13312x5120 | 16 | i8a8 | 93.7 [93.6–93.8] | 93.9 [93.8–94.2] | +0.2% |
| 13312x5120 | 20 | iq8r | 136.9 [136.7–137.4] | 143.3 [142.6–143.4] | +4.7% |
| 13312x5120 | 20 | i8a8 | 138.3 [137.8–139.9] | 148.4 [148.2–148.7] | +7.3% |
| 13312x5120 | 24 | iq8r | 137.2 [137.2–137.3] | 143.5 [143.4–143.6] | +4.6% |
| 13312x5120 | 24 | i8a8 | 139.9 [139.5–140.0] | 148.8 [148.8–148.9] | +6.4% |
| 13312x5120 | 28 | iq8r | 137.5 [137.2–137.8] | 143.9 [143.8–144.0] | +4.7% |
| 13312x5120 | 28 | i8a8 | 140.0 [139.6–140.4] | 149.0 [148.8–149.1] | +6.4% |
| 13312x5120 | 32 | iq8r | 137.4 [137.3–137.5] | 144.2 [143.9–144.2] | +5.0% |
| 13312x5120 | 32 | i8a8 | 140.8 [140.8–141.1] | 149.7 [149.4–149.8] | +6.3% |
| 13312x5120 | 40 | iq8r | 190.7 [190.5–190.8] | 206.6 [206.2–206.7] | +8.3% |
| 13312x5120 | 40 | i8a8 | 193.9 [193.6–195.1] | 212.9 [212.6–213.1] | +9.8% |
| 13312x5120 | 48 | iq8r | 191.5 [191.1–191.6] | 209.1 [208.2–209.2] | +9.2% |
| 13312x5120 | 48 | i8a8 | 196.1 [195.5–196.2] | 215.8 [215.1–215.8] | +10.0% |
| 5120x6656 | 4 | iq8r | 55.6 [55.4–55.9] | 55.4 [55.3–55.9] | -0.2% |
| 5120x6656 | 4 | i8a8 | 57.7 [56.4–59.4] | 58.2 [58.1–60.2] | +0.8% |
| 5120x6656 | 16 | iq8r | 55.6 [55.5–56.5] | 55.8 [55.8–56.0] | +0.4% |
| 5120x6656 | 16 | i8a8 | 58.8 [58.6–59.1] | 59.0 [58.8–59.0] | +0.3% |
| 5120x6656 | 20 | iq8r | 86.1 [86.0–86.1] | 82.9 [82.8–83.0] | -3.8% |
| 5120x6656 | 20 | i8a8 | 83.1 [83.0–83.2] | 82.8 [81.3–83.7] | -0.3% |
| 5120x6656 | 24 | iq8r | 86.4 [86.3–86.5] | 83.1 [82.9–83.2] | -3.8% |
| 5120x6656 | 24 | i8a8 | 83.1 [83.1–83.2] | 84.0 [83.9–84.2] | +1.1% |
| 5120x6656 | 28 | iq8r | 86.5 [86.4–86.6] | 83.2 [83.0–83.2] | -3.8% |
| 5120x6656 | 28 | i8a8 | 83.3 [83.2–83.3] | 84.4 [84.2–84.4] | +1.3% |
| 5120x6656 | 32 | iq8r | 86.8 [86.8–86.8] | 83.1 [83.1–83.4] | -4.2% |
| 5120x6656 | 32 | i8a8 | 83.3 [83.2–83.4] | 84.5 [84.1–84.6] | +1.4% |
| 5120x6656 | 40 | iq8r | 114.5 [114.2–115.0] | 116.8 [116.4–117.0] | +2.0% |
| 5120x6656 | 40 | i8a8 | 110.8 [110.6–110.8] | 116.5 [116.2–116.9] | +5.1% |
| 5120x6656 | 48 | iq8r | 115.2 [115.2–115.3] | 117.0 [116.8–117.2] | +1.6% |
| 5120x6656 | 48 | i8a8 | 111.0 [110.9–111.0] | 117.1 [117.0–117.1] | +5.6% |
| 6144x5120 | 4 | iq8r | 50.7 [50.6–50.7] | 50.8 [50.7–51.0] | +0.2% |
| 6144x5120 | 4 | i8a8 | 50.7 [50.6–52.8] | 50.6 [50.4–52.3] | -0.2% |
| 6144x5120 | 16 | iq8r | 51.0 [50.7–51.1] | 51.0 [50.9–51.3] | +0.1% |
| 6144x5120 | 16 | i8a8 | 50.8 [50.8–51.3] | 51.0 [51.0–51.1] | +0.4% |
| 6144x5120 | 20 | iq8r | 67.5 [67.3–67.5] | 68.5 [68.3–68.6] | +1.5% |
| 6144x5120 | 20 | i8a8 | 68.0 [67.6–69.1] | 78.5 [78.5–78.7] | +15.5% |
| 6144x5120 | 24 | iq8r | 67.5 [67.4–67.5] | 68.6 [68.5–68.7] | +1.6% |
| 6144x5120 | 24 | i8a8 | 68.5 [67.8–70.0] | 78.6 [78.5–78.6] | +14.7% |
| 6144x5120 | 28 | iq8r | 67.8 [67.4–67.8] | 68.9 [68.7–69.0] | +1.7% |
| 6144x5120 | 28 | i8a8 | 68.4 [68.3–78.5] | 78.6 [78.6–78.7] | +15.0% |
| 6144x5120 | 32 | iq8r | 67.6 [67.5–67.7] | 69.5 [69.3–70.8] | +2.8% |
| 6144x5120 | 32 | i8a8 | 70.0 [69.2–78.5] | 78.8 [78.8–78.9] | +12.7% |
| 6144x5120 | 40 | iq8r | 97.4 [97.1–97.8] | 100.6 [99.6–101.1] | +3.3% |
| 6144x5120 | 40 | i8a8 | 99.8 [99.6–99.8] | 104.1 [104.0–104.2] | +4.2% |
| 6144x5120 | 48 | iq8r | 97.6 [97.6–97.8] | 101.9 [101.9–102.1] | +4.4% |
| 6144x5120 | 48 | i8a8 | 100.0 [99.9–100.0] | 104.3 [104.2–104.6] | +4.3% |
| 5120x2304 | 4 | iq8r | 28.2 [28.0–28.4] | 29.1 [28.2–29.5] | +3.1% |
| 5120x2304 | 4 | i8a8 | 28.9 [28.5–30.4] | 28.8 [27.0–30.5] | -0.4% |
| 5120x2304 | 16 | iq8r | 29.1 [28.4–29.2] | 28.4 [28.1–29.8] | -2.5% |
| 5120x2304 | 16 | i8a8 | 26.9 [26.8–26.9] | 26.8 [26.8–26.9] | -0.1% |
| 5120x2304 | 20 | iq8r | 41.1 [41.0–41.3] | 37.8 [37.8–37.9] | -7.9% |
| 5120x2304 | 20 | i8a8 | 37.3 [37.2–37.3] | 34.9 [34.9–35.0] | -6.3% |
| 5120x2304 | 24 | iq8r | 41.4 [41.3–41.4] | 37.9 [37.9–38.0] | -8.4% |
| 5120x2304 | 24 | i8a8 | 37.4 [37.3–37.4] | 35.0 [34.9–35.1] | -6.4% |
| 5120x2304 | 28 | iq8r | 41.3 [41.3–41.4] | 38.2 [38.2–38.2] | -7.6% |
| 5120x2304 | 28 | i8a8 | 37.5 [37.4–37.6] | 35.1 [34.8–35.2] | -6.4% |
| 5120x2304 | 32 | iq8r | 41.3 [41.2–41.4] | 38.1 [37.9–38.2] | -7.7% |
| 5120x2304 | 32 | i8a8 | 37.4 [37.4–37.5] | 35.1 [35.0–35.2] | -6.2% |
| 5120x2304 | 40 | iq8r | 52.4 [52.4–52.5] | 50.1 [50.0–50.2] | -4.4% |
| 5120x2304 | 40 | i8a8 | 47.7 [47.5–47.8] | 45.6 [45.6–45.6] | -4.4% |
| 5120x2304 | 48 | iq8r | 52.7 [52.6–52.8] | 50.3 [50.2–50.4] | -4.6% |
| 5120x2304 | 48 | i8a8 | 47.8 [47.6–47.9] | 46.0 [45.8–46.0] | -3.9% |
| 3072x5120 | 4 | iq8r | 27.9 [27.8–28.0] | 27.8 [27.8–27.8] | -0.4% |
| 3072x5120 | 4 | i8a8 | 27.5 [27.4–32.0] | 27.3 [27.3–32.3] | -0.6% |
| 3072x5120 | 16 | iq8r | 27.9 [27.9–28.1] | 28.1 [28.0–28.2] | +0.6% |
| 3072x5120 | 16 | i8a8 | 27.5 [27.4–27.6] | 27.7 [27.7–27.7] | +0.9% |
| 3072x5120 | 20 | iq8r | 37.5 [37.3–38.4] | 41.0 [40.9–41.2] | +9.4% |
| 3072x5120 | 20 | i8a8 | 40.4 [40.4–40.5] | 42.4 [42.3–42.4] | +5.0% |
| 3072x5120 | 24 | iq8r | 39.2 [37.2–39.9] | 41.4 [41.4–41.4] | +5.7% |
| 3072x5120 | 24 | i8a8 | 40.5 [40.4–40.5] | 42.4 [42.4–42.5] | +4.7% |
| 3072x5120 | 28 | iq8r | 39.8 [37.6–40.0] | 41.6 [41.4–41.6] | +4.4% |
| 3072x5120 | 28 | i8a8 | 40.6 [40.4–40.6] | 42.5 [42.4–42.5] | +4.5% |
| 3072x5120 | 32 | iq8r | 39.8 [39.4–40.1] | 41.8 [41.7–41.8] | +4.9% |
| 3072x5120 | 32 | i8a8 | 40.6 [40.5–40.6] | 42.7 [42.7–42.9] | +5.3% |
| 3072x5120 | 40 | iq8r | 53.9 [53.9–54.0] | 52.9 [52.5–53.0] | -1.9% |
| 3072x5120 | 40 | i8a8 | 55.5 [55.4–55.6] | 54.2 [54.2–54.4] | -2.2% |
| 3072x5120 | 48 | iq8r | 54.4 [54.4–54.5] | 53.4 [53.4–54.1] | -1.9% |
| 3072x5120 | 48 | i8a8 | 55.5 [55.3–55.5] | 55.9 [55.7–56.3] | +0.7% |
| 5120x1536 | 4 | iq8r | 21.5 [21.5–21.6] | 21.4 [21.4–21.6] | -0.4% |
| 5120x1536 | 4 | i8a8 | 21.2 [21.1–22.8] | 22.0 [20.9–23.6] | +4.0% |
| 5120x1536 | 16 | iq8r | 21.6 [21.5–21.6] | 21.5 [21.5–21.6] | -0.6% |
| 5120x1536 | 16 | i8a8 | 21.8 [21.4–21.8] | 21.6 [21.5–21.8] | -0.9% |
| 5120x1536 | 20 | iq8r | 32.2 [32.0–32.3] | 27.3 [27.3–27.4] | -15.2% |
| 5120x1536 | 20 | i8a8 | 28.5 [28.3–28.6] | 25.8 [25.8–25.9] | -9.4% |
| 5120x1536 | 24 | iq8r | 32.2 [32.2–32.3] | 27.6 [27.5–27.6] | -14.5% |
| 5120x1536 | 24 | i8a8 | 28.4 [28.4–28.6] | 26.0 [26.0–26.1] | -8.4% |
| 5120x1536 | 28 | iq8r | 32.2 [32.2–32.3] | 27.5 [27.4–27.6] | -14.6% |
| 5120x1536 | 28 | i8a8 | 28.4 [28.4–28.4] | 25.8 [25.8–25.9] | -9.2% |
| 5120x1536 | 32 | iq8r | 32.2 [32.0–32.3] | 27.6 [27.6–27.7] | -14.3% |
| 5120x1536 | 32 | i8a8 | 28.4 [28.4–28.6] | 25.9 [25.7–26.0] | -9.0% |
| 5120x1536 | 40 | iq8r | 40.4 [40.3–40.4] | 35.8 [35.7–35.9] | -11.4% |
| 5120x1536 | 40 | i8a8 | 35.4 [35.2–35.5] | 34.1 [34.0–34.2] | -3.7% |
| 5120x1536 | 48 | iq8r | 40.3 [40.3–40.4] | 36.4 [36.0–37.5] | -9.8% |
| 5120x1536 | 48 | i8a8 | 35.4 [35.4–35.4] | 34.3 [34.3–34.3] | -3.2% |

## Measured acceptance gates

**1. Cold effective bandwidth: FAIL.** Unique int8 weight bytes plus BF16 weight scales divided by kernel time (not a hardware DRAM counter). At M=32 the NT2 candidate must retain >=80% of the same row's 2020.8.1 NT1 throughput; the four tested combinations miss:

| N×K | Row | NT1 baseline GB/s | NT2 candidate GB/s | Ratio |
|---|---|---:|---:|---:|
| 13312x5120 | iq8r | 734.5 | 476.5 | 64.9% (FAIL) |
| 13312x5120 | i8a8 | 738.6 | 462.3 | 62.6% (FAIL) |
| 5120x6656 | iq8r | 617.3 | 413.2 | 66.9% (FAIL) |
| 5120x6656 | i8a8 | 588.6 | 409.7 | 69.6% (FAIL) |

The effective-bandwidth ratio implies extra work or traffic, but cannot uniquely attribute bytes to DRAM without counters. It is not proof that the duplicate loads missed in cache.

**2. LDS cross-wave reduction penalty: NOT RUN.** A scratch-only, wrong-numerics reduction-skip NT2 debug variant was prepared, but its compilation was canceled after Gate 1 failed. No debug GPU measurement, so the <5% bound is unverified. No debug variant was committed.

**3. VGPR/occupancy: PASS.** All records and waves-per-SIMD are above. The NT1 records and 278 unrelated GEMM-unit resource records remain unchanged, while NT2 uses more waves per block but fewer registers per lane.

**4. Small grids: MEASURED, MIXED.** The original N/16 grid remains: 3072×5120 has 192 blocks, 5120×1536 has 320 blocks, NT2 uses 8 waves per block for KW4. M32 IQ8R: 39.8→41.8 µs (+4.9%) on 192 blocks, versus 32.2→27.6 µs (-14.3%) on 320; I8A8: 40.6→42.7 (+5.3%) versus 28.4→25.9 (-9.0%). No universal starvation or improvement conclusion.

**Latency acceptance: FAIL.** 13312×5120 IQ8R at M20/24/28/32 is 143.3–144.2 µs (2020.8.1 is 136.9–137.5), failing the strict below-137.3-µs target and the 93–137-µs window; expected 100–115 µs not met. I8A8 at the same M values is 148.4–149.7 vs baseline 138.3–140.8 µs. NT1 M4 and M16 remain within small noise; NT3 at M40 and 48 becomes slower on both large shapes (IQ8R 190.7→206.6 µs and 191.5→209.1 µs on gate-up, 114.5→116.8 and 115.2→117.0 on down). NT4 M49–64 was bit/resource tested on eligible shapes, but deliberately not cold-replayed (M selection stopped at 48).

## Explicit executed versus omitted tests

- Executed: rewrite/CMake apply, **single GEMM-unit** gfx1100 libr3 compile and link (R3_ONLY=r4d_gemm_fp8a8), gfx1100 notes on all 48 K-parallel instantiations and 278 other GEMM-unit kernels, same-process BF16 identity (2,304 pass), all six-shape cold replays (288 paired records), and git diff --check. All GPU executions used a separate queued 12m TTL/5m estimate lease and released it on exit. An initial cold replay command failed to load a ROCm shared library before any GPU launch; correction of the library search path produced all 288 expected records.
- Not executed: **full libr3 plugin build**, **r3_selftest target build**, **r3_selftest narrow/IQR checks** (no zero-FAIL claim), debug reduction-skip timing, serving or generation, ms/step A/B, KLD, shipping, pushing, or merging. Stop justified by Gate 1 failure.
- Remote plugin review: checked the one-file native rewrite and changed scope, checked all GEMM-unit non-KPAR metadata records unchanged, reviewed performance rejection and diff whitespace. Maintained code increases and misses the mandatory performance bound; lead should not ship this candidate.

## Reproducer

Build with cmake -S engines/radiance/components/libr3 -B <scratch>/build-gemm -G Ninja -DCMAKE_PREFIX_PATH=<runtime-prefix> -DRADIANCE_SRC=<pinned-source> -DR3_TARGETS=gfx1100 -DR3_NATIVE=ON -DR3_ONLY=r4d_gemm_fp8a8; then cmake --build <scratch>/build-gemm --target libr3 -j2. Extract gfx1100 code objects with llvm-objdump --offloading and inspect llvm-readelf -n. Run <bitcheck-iq> and <bitcheck-i8> with <release-2020.8.1-plugin> <candidate-plugin>, with and without CHAIN_TEST=1. All GPU runs use agent-lease acquire gpu --ttl 12m --est 5m --wait -- env HIP_VISIBLE_DEVICES=0 <command>. For cold replay run the standalone <cold-replay> <plugin> <N> <K> <before-or-after>, with ROCm shared libraries in LD_LIBRARY_PATH, over each of the six shapes and both arms, each in its own lease. Scratch sources and all 288 raw JSONL records were retained outside the repository.
