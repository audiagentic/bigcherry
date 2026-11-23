# VTP37 — K-parallel accumulator-conversion-once experiment

**Decision: measured and accepted for shipping.** The full gfx1100 libr3 plugin builds with all 115 units; the required narrow-row/IQR r3_selftest checks, resource gate, exact BF16 byte identity and cold replay pass. No serving, ms/step A/B or KLD test was run; these remain a downstream decision.

## Scope and implementation

Only the native gfx1100 rewrite of `r4d_gemm_fp8a8.hip` changed: inside `gemm_i8_kpar_body`, keep each IU8 WMMA accumulator in gfx1100 `r3_i8` layout across all 8/16 operations in one K scale group. The operands still use `r3_bytes_12_to_11`; after the last WMMA of each group, convert each token tile once with `r3_acc_i32_11_to_12` before the existing BF16-scale fold. The order of every integer sum, scale multiplication, fmaf, LDS reduction and output store remains as previously. The SB256 `#pragma unroll ((SB == 256u && NT >= 2u) ? 1u : kUG)` rule from #1442 remains unchanged.

No IQR source/rewrite, vendor source, gfx1201 path, generic compat layer, narrow-tile implementation or prefill tile was edited. The rewrite drift check returned `{"failures": {}}`, and `git diff --check` passed.

## Full build and required r3_selftest (actually executed)

Configured the entire libr3 gfx1100 plugin (R3_NATIVE=ON; **R3_ONLY unset**), 115 of 115 device units, 0 excluded. `nice -n 19 cmake --build <build> --target libr3 r3_selftest -j 3` completed 132/132 Ninja steps and exit 0, including `r4d_gemm_fp8a8.hip`, IQR modules and the `r3_selftest` executable. Link reported **zero stubbed libr4d symbols**. Nonfatal compiler warnings were emitted (including the pre-existing #1442 unroll-pragma compatibility warning and unrelated occupancy warnings).

The full-plugin `r3_selftest <full-plugin.so> --case CASE` executable was run independently for each of these **five** targeted rows on a queued short GPU lease:

| Exact r3_selftest --case | Passing checks | FAIL | SKIP |
|---|---:|---:|---:|
| gemm_i8a8_nt_m16 | 13 | 0 | 0 |
| gemm_iq8r_nt_m16 | 35 | 0 | 0 |
| gemm_iqr_nt_m64 | 1032 | 0 | 0 |
| gemm_iqr_tiled | 129 | 0 | 0 |
| gemm_iqr_gated | 36 | 0 | 0 |

**Total: 1,245 passing `ok` checks; zero `FAIL`, zero `SKIP`, all five processes returned 0.** These were the *targeted* narrow-row/IQR checks, not the entire all-row selftest catalogue.

## Gfx1100 code-object resource table

Extracted the GEMM code object from the **full linked plugin** using `llvm-objdump --offloading` and inspected AMDGPU kernel metadata with ROCm's `llvm-readobj --notes`. Resource cells: VGPR / VGPR spills / private bytes / floor(1536/VGPR) theoretical wave32 waves per SIMD. Every KW2 and KW4, SB128 and SB256, NT1–4, chained and unchained dispatch is below:

| KW | SB | NT | mode | baseline (VGPR/spill/private/waves) | candidate (VGPR/spill/private/waves) |
|---:|---:|---:|---|---|---|
| 2 | 128 | 1 | chained | 151/0/0/10 | 144/0/0/10 |
| 2 | 128 | 1 | unchained | 148/0/0/10 | 141/0/0/10 |
| 2 | 128 | 2 | chained | 224/0/0/6 | 178/0/0/8 |
| 2 | 128 | 2 | unchained | 221/0/0/6 | 175/0/0/8 |
| 2 | 128 | 3 | chained | 240/3/16/6 | 240/3/16/6 |
| 2 | 128 | 3 | unchained | 240/2/12/6 | 240/1/8/6 |
| 2 | 128 | 4 | chained | 240/43/120/6 | 180/0/0/8 |
| 2 | 128 | 4 | unchained | 240/35/96/6 | 177/0/0/8 |
| 2 | 256 | 1 | chained | 203/0/0/7 | 213/0/0/7 |
| 2 | 256 | 1 | unchained | 200/0/0/7 | 210/0/0/7 |
| 2 | 256 | 2 | chained | 181/0/0/8 | 177/0/0/8 |
| 2 | 256 | 2 | unchained | 178/0/0/8 | 174/0/0/8 |
| 2 | 256 | 3 | chained | 210/0/0/7 | 215/0/0/7 |
| 2 | 256 | 3 | unchained | 207/0/0/7 | 212/0/0/7 |
| 2 | 256 | 4 | chained | 234/0/0/6 | 174/0/0/8 |
| 2 | 256 | 4 | unchained | 231/0/0/6 | 171/0/0/8 |
| 4 | 128 | 1 | chained | 151/0/0/10 | 144/0/0/10 |
| 4 | 128 | 1 | unchained | 148/0/0/10 | 141/0/0/10 |
| 4 | 128 | 2 | chained | 224/0/0/6 | 178/0/0/8 |
| 4 | 128 | 2 | unchained | 221/0/0/6 | 175/0/0/8 |
| 4 | 128 | 3 | chained | 240/3/16/6 | 240/3/16/6 |
| 4 | 128 | 3 | unchained | 240/2/12/6 | 240/1/8/6 |
| 4 | 128 | 4 | chained | 240/43/120/6 | 180/0/0/8 |
| 4 | 128 | 4 | unchained | 240/35/96/6 | 177/0/0/8 |
| 4 | 256 | 1 | chained | 203/0/0/7 | 213/0/0/7 |
| 4 | 256 | 1 | unchained | 200/0/0/7 | 210/0/0/7 |
| 4 | 256 | 2 | chained | 181/0/0/8 | 177/0/0/8 |
| 4 | 256 | 2 | unchained | 178/0/0/8 | 174/0/0/8 |
| 4 | 256 | 3 | chained | 210/0/0/7 | 215/0/0/7 |
| 4 | 256 | 3 | unchained | 207/0/0/7 | 212/0/0/7 |
| 4 | 256 | 4 | chained | 234/0/0/6 | 174/0/0/8 |
| 4 | 256 | 4 | unchained | 231/0/0/6 | 171/0/0/8 |

All 48 K-parallel symbols (including KW1) match between the 2020.8.1 baseline and full candidate; **zero new spills**, zero increased private scratch and **zero declines in theoretical waves/SIMD**. Across the 278 non-K-parallel symbols in this GEMM device object, all 278 resource records are identical, with zero symbols added or removed. In particular, SB128 NT2 unchained VGPR falls 221→175 (6→8 theoretical waves); SB256 NT2 falls 178→174 (8 unchanged). SB256 NT1 increases 200→210 VGPR (7 waves unchanged, no spills). SB128 NT3 chained retains three pre-existing VGPR spills and 16 private bytes, whereas SB128 NT4 removes its 35–43 spill registers; this gate prohibits new spills, not pre-existing ones.

## Exact BF16 output bits (full linked plugin)

For each row and dispatch mode, the full candidate and 2020.8.1 baseline plugin were loaded together and compared byte-for-byte over all output BF16 values. All six drafter shapes were checked at **every M 1–64**, plus a ragged N=5136, K=5120 case and KW2 short-K tests at (N,K)=(5120,512) and (5120,256). Inputs and scale tables were deterministic and identical; destination-touch checks were enabled.

| Row / dispatch | PASS | SKIP | FAIL |
|---|---:|---:|---:|
| iq-unchained | 576 | 0 | 0 |
| iq-chained | 576 | 0 | 0 |
| i8-unchained | 576 | 0 | 0 |
| i8-chained | 576 | 0 | 0 |

**2,304/2,304 passed, zero skips and zero failures.** Chained/unchained launch-parameter dispatch was checked; this was *not* a serving chain-signal integration test. An isolated one-unit candidate was also tested and returned the same 2,304/2,304 pass total; only full-plugin results are counted in the acceptance gate.

## Cold replay against the full linked plugin

One gfx1100 device, 256 MiB D2D cache flush before every timed launch, rotated int8 weight/scale copies, 30 launches per measurement, three repetitions, one **independent queued GPU lease per shape/arm**. Both rows are measured in each shape/arm run. Six shapes × eight M values [4,16,20,24,28,32,40,48] × two arms × three repeats = **288 raw JSONL records**, giving 96 paired row/M/shape comparisons. Values are median [min-max] of three repeats, in microseconds; negative deltas mean faster candidate.

| Shape | M | Row | Before us [min-max] | After us [min-max] | Delta % |
|---|---:|---|---:|---:|---:|
| 13312x5120 | 4 | iq8r | 91.2 [91.1-91.2] | 89.7 [89.4-89.8] | -1.6% |
| 13312x5120 | 4 | i8a8 | 91.6 [91.6-96.3] | 90.3 [90.2-92.2] | -1.5% |
| 13312x5120 | 16 | iq8r | 93.2 [93.2-93.3] | 91.3 [91.2-91.4] | -2.1% |
| 13312x5120 | 16 | i8a8 | 93.8 [93.8-94.2] | 92.4 [92.4-92.6] | -1.5% |
| 13312x5120 | 20 | iq8r | 136.9 [136.8-137.1] | 96.8 [96.7-96.9] | -29.3% |
| 13312x5120 | 20 | i8a8 | 138.3 [138.0-138.9] | 98.8 [98.6-98.9] | -28.6% |
| 13312x5120 | 24 | iq8r | 137.1 [137.0-137.4] | 97.8 [97.6-97.8] | -28.7% |
| 13312x5120 | 24 | i8a8 | 140.0 [138.6-140.3] | 99.8 [99.3-99.8] | -28.7% |
| 13312x5120 | 28 | iq8r | 137.4 [137.1-137.4] | 98.3 [98.2-98.4] | -28.4% |
| 13312x5120 | 28 | i8a8 | 140.7 [140.7-140.9] | 100.1 [100.1-100.8] | -28.9% |
| 13312x5120 | 32 | iq8r | 137.5 [137.5-137.8] | 98.8 [98.5-99.0] | -28.2% |
| 13312x5120 | 32 | i8a8 | 141.2 [140.9-141.3] | 100.7 [100.6-101.1] | -28.7% |
| 13312x5120 | 40 | iq8r | 190.7 [190.4-191.0] | 113.7 [113.6-113.9] | -40.4% |
| 13312x5120 | 40 | i8a8 | 195.8 [195.1-195.9] | 120.7 [120.6-120.8] | -38.4% |
| 13312x5120 | 48 | iq8r | 192.5 [192.1-192.6] | 114.6 [114.5-114.6] | -40.5% |
| 13312x5120 | 48 | i8a8 | 196.2 [196.1-196.3] | 121.6 [121.3-121.7] | -38.0% |
| 5120x6656 | 4 | iq8r | 55.5 [55.4-55.7] | 51.2 [51.1-51.2] | -7.7% |
| 5120x6656 | 4 | i8a8 | 58.1 [57.8-59.9] | 50.1 [49.8-54.2] | -13.7% |
| 5120x6656 | 16 | iq8r | 55.9 [55.2-56.2] | 51.1 [50.8-51.2] | -8.5% |
| 5120x6656 | 16 | i8a8 | 58.9 [58.4-59.0] | 50.2 [50.2-50.3] | -14.7% |
| 5120x6656 | 20 | iq8r | 86.2 [86.0-86.3] | 57.4 [57.1-57.6] | -33.3% |
| 5120x6656 | 20 | i8a8 | 83.0 [82.9-83.0] | 54.0 [53.5-54.2] | -34.9% |
| 5120x6656 | 24 | iq8r | 86.3 [86.0-86.5] | 57.9 [57.5-58.1] | -32.9% |
| 5120x6656 | 24 | i8a8 | 83.1 [82.8-83.3] | 53.6 [53.6-53.9] | -35.5% |
| 5120x6656 | 28 | iq8r | 86.6 [86.6-86.6] | 58.4 [58.2-58.4] | -32.6% |
| 5120x6656 | 28 | i8a8 | 83.2 [83.1-83.4] | 53.7 [53.6-54.2] | -35.4% |
| 5120x6656 | 32 | iq8r | 86.5 [86.5-86.7] | 58.3 [57.6-58.6] | -32.6% |
| 5120x6656 | 32 | i8a8 | 83.2 [83.1-83.4] | 54.6 [54.4-54.7] | -34.4% |
| 5120x6656 | 40 | iq8r | 114.6 [114.5-115.0] | 65.2 [65.2-65.3] | -43.1% |
| 5120x6656 | 40 | i8a8 | 110.8 [110.7-110.8] | 66.1 [65.8-66.2] | -40.4% |
| 5120x6656 | 48 | iq8r | 115.4 [114.9-115.7] | 65.3 [65.2-65.5] | -43.4% |
| 5120x6656 | 48 | i8a8 | 110.9 [110.7-111.0] | 65.9 [65.7-66.0] | -40.6% |
| 6144x5120 | 4 | iq8r | 50.9 [50.6-51.0] | 45.2 [44.9-45.4] | -11.2% |
| 6144x5120 | 4 | i8a8 | 50.5 [50.5-52.1] | 44.4 [44.2-47.6] | -12.2% |
| 6144x5120 | 16 | iq8r | 51.1 [50.8-51.3] | 45.4 [45.4-45.4] | -11.3% |
| 6144x5120 | 16 | i8a8 | 50.7 [50.6-50.9] | 45.1 [44.9-45.2] | -11.1% |
| 6144x5120 | 20 | iq8r | 67.4 [67.4-67.4] | 49.9 [49.6-49.9] | -26.0% |
| 6144x5120 | 20 | i8a8 | 68.0 [67.7-68.2] | 48.1 [48.0-48.1] | -29.3% |
| 6144x5120 | 24 | iq8r | 67.5 [67.4-67.5] | 50.2 [50.1-50.3] | -25.7% |
| 6144x5120 | 24 | i8a8 | 68.3 [67.8-68.3] | 48.2 [48.2-48.3] | -29.4% |
| 6144x5120 | 28 | iq8r | 67.6 [67.4-67.8] | 50.4 [50.2-50.5] | -25.4% |
| 6144x5120 | 28 | i8a8 | 68.7 [67.8-69.4] | 48.4 [48.4-48.4] | -29.5% |
| 6144x5120 | 32 | iq8r | 67.6 [67.6-67.8] | 50.4 [50.3-50.5] | -25.5% |
| 6144x5120 | 32 | i8a8 | 70.1 [68.8-70.3] | 48.6 [48.6-48.8] | -30.6% |
| 6144x5120 | 40 | iq8r | 97.6 [97.2-97.6] | 57.5 [57.4-57.5] | -41.0% |
| 6144x5120 | 40 | i8a8 | 99.8 [99.8-99.9] | 60.6 [60.5-60.6] | -39.3% |
| 6144x5120 | 48 | iq8r | 97.7 [97.6-97.9] | 57.6 [57.4-57.6] | -41.0% |
| 6144x5120 | 48 | i8a8 | 99.8 [99.8-99.8] | 60.8 [60.7-60.9] | -39.1% |
| 5120x2304 | 4 | iq8r | 28.1 [28.0-29.2] | 24.5 [24.5-24.6] | -12.9% |
| 5120x2304 | 4 | i8a8 | 28.5 [28.3-29.2] | 23.3 [23.2-25.8] | -18.4% |
| 5120x2304 | 16 | iq8r | 28.3 [28.1-29.6] | 24.5 [24.4-24.5] | -13.6% |
| 5120x2304 | 16 | i8a8 | 27.2 [27.1-27.7] | 23.4 [23.4-23.6] | -14.2% |
| 5120x2304 | 20 | iq8r | 41.1 [41.1-41.1] | 30.6 [30.4-30.8] | -25.5% |
| 5120x2304 | 20 | i8a8 | 37.2 [37.2-37.3] | 28.4 [28.3-28.5] | -23.6% |
| 5120x2304 | 24 | iq8r | 41.4 [41.2-41.5] | 30.7 [30.6-30.8] | -25.9% |
| 5120x2304 | 24 | i8a8 | 37.4 [37.2-37.5] | 28.4 [28.1-28.4] | -24.1% |
| 5120x2304 | 28 | iq8r | 41.3 [41.1-41.5] | 30.8 [30.8-30.9] | -25.4% |
| 5120x2304 | 28 | i8a8 | 37.5 [37.4-37.5] | 28.2 [28.0-28.4] | -24.9% |
| 5120x2304 | 32 | iq8r | 41.2 [41.2-41.2] | 30.7 [30.5-30.8] | -25.5% |
| 5120x2304 | 32 | i8a8 | 37.5 [37.4-37.5] | 27.9 [27.8-28.0] | -25.6% |
| 5120x2304 | 40 | iq8r | 52.5 [52.4-52.6] | 34.6 [34.6-34.6] | -34.1% |
| 5120x2304 | 40 | i8a8 | 47.7 [47.6-47.7] | 32.9 [32.9-33.0] | -31.0% |
| 5120x2304 | 48 | iq8r | 52.6 [52.6-52.8] | 34.7 [34.7-34.8] | -34.0% |
| 5120x2304 | 48 | i8a8 | 47.8 [47.8-47.9] | 33.0 [33.0-33.2] | -31.0% |
| 3072x5120 | 4 | iq8r | 28.0 [27.7-28.0] | 26.4 [26.2-26.4] | -5.7% |
| 3072x5120 | 4 | i8a8 | 27.4 [27.4-32.8] | 26.3 [26.2-28.8] | -4.1% |
| 3072x5120 | 16 | iq8r | 27.8 [27.8-28.1] | 26.4 [26.4-26.5] | -4.9% |
| 3072x5120 | 16 | i8a8 | 27.5 [27.3-27.6] | 26.5 [26.2-26.6] | -3.6% |
| 3072x5120 | 20 | iq8r | 38.6 [37.2-38.9] | 28.3 [28.2-28.4] | -26.7% |
| 3072x5120 | 20 | i8a8 | 40.4 [40.4-40.5] | 28.0 [27.9-28.2] | -30.9% |
| 3072x5120 | 24 | iq8r | 38.2 [37.4-39.3] | 28.4 [28.4-28.5] | -25.7% |
| 3072x5120 | 24 | i8a8 | 40.4 [40.4-40.4] | 28.2 [28.0-28.6] | -30.3% |
| 3072x5120 | 28 | iq8r | 39.6 [37.3-39.6] | 28.6 [28.4-28.8] | -27.6% |
| 3072x5120 | 28 | i8a8 | 40.6 [40.4-40.6] | 28.5 [28.2-28.5] | -29.9% |
| 3072x5120 | 32 | iq8r | 40.0 [39.9-40.1] | 28.7 [28.7-28.8] | -28.2% |
| 3072x5120 | 32 | i8a8 | 40.5 [40.5-40.6] | 28.4 [28.3-28.4] | -29.8% |
| 3072x5120 | 40 | iq8r | 54.0 [53.6-54.2] | 33.3 [33.2-33.4] | -38.3% |
| 3072x5120 | 40 | i8a8 | 55.3 [55.2-55.4] | 35.0 [34.9-35.1] | -36.8% |
| 3072x5120 | 48 | iq8r | 54.4 [54.2-54.5] | 33.7 [33.6-33.8] | -38.0% |
| 3072x5120 | 48 | i8a8 | 55.6 [55.5-55.6] | 35.1 [35.0-35.1] | -36.9% |
| 5120x1536 | 4 | iq8r | 21.7 [21.6-21.7] | 18.9 [18.8-19.0] | -12.7% |
| 5120x1536 | 4 | i8a8 | 21.2 [21.0-22.3] | 18.2 [18.1-19.6] | -13.8% |
| 5120x1536 | 16 | iq8r | 21.7 [21.6-21.7] | 19.0 [19.0-19.1] | -12.2% |
| 5120x1536 | 16 | i8a8 | 22.1 [22.1-22.2] | 18.2 [18.2-18.2] | -17.8% |
| 5120x1536 | 20 | iq8r | 32.3 [32.0-32.5] | 23.8 [23.7-23.9] | -26.2% |
| 5120x1536 | 20 | i8a8 | 28.6 [28.3-28.6] | 21.7 [21.6-22.2] | -24.2% |
| 5120x1536 | 24 | iq8r | 32.3 [32.1-32.3] | 23.9 [23.8-23.9] | -26.1% |
| 5120x1536 | 24 | i8a8 | 28.5 [28.5-28.6] | 22.0 [21.9-22.3] | -22.6% |
| 5120x1536 | 28 | iq8r | 32.3 [32.2-32.4] | 23.8 [23.6-24.0] | -26.3% |
| 5120x1536 | 28 | i8a8 | 28.6 [28.4-28.6] | 22.0 [21.9-22.1] | -23.1% |
| 5120x1536 | 32 | iq8r | 32.4 [32.2-32.4] | 23.8 [23.8-23.8] | -26.5% |
| 5120x1536 | 32 | i8a8 | 28.6 [28.5-28.6] | 22.1 [22.0-22.2] | -22.5% |
| 5120x1536 | 40 | iq8r | 40.5 [40.4-40.6] | 28.1 [28.0-28.2] | -30.5% |
| 5120x1536 | 40 | i8a8 | 35.5 [35.5-35.5] | 26.0 [26.0-26.0] | -26.8% |
| 5120x1536 | 48 | iq8r | 40.5 [40.4-40.5] | 28.3 [28.2-28.4] | -30.1% |
| 5120x1536 | 48 | i8a8 | 35.4 [35.3-35.4] | 26.1 [26.1-26.2] | -26.3% |

**All 96 measured comparisons improve** (smallest improvement 1.5%, largest 43.4%). No small-M (4,16) slowdown or measured NT3 (M40,48) regression. M49–64 (NT4) was bit-checked and resource-checked but **not cold-timed**, since the specified timing replay stops at M48.

### Gate-up acceptance, M20/24/28/32

| M | IQ8R before → after us (improvement) | I8A8 before → after us (improvement) |
|---:|---|---|
| 20 | 136.9 → 96.8 (29.3% faster) | 138.3 → 98.8 (28.6% faster) |
| 24 | 137.1 → 97.8 (28.7% faster) | 140.0 → 99.8 (28.7% faster) |
| 28 | 137.4 → 98.3 (28.4% faster) | 140.7 → 100.1 (28.9% faster) |
| 32 | 137.5 → 98.8 (28.2% faster) | 141.2 → 100.7 (28.7% faster) |

**Gate: PASS.** All eight gate-up row/M comparisons exceed the required 20% improvement; IQ8R M20–32 lands between 96.8 and 98.8 µs and the other row between 98.8 and 100.7 µs.

### Effective bandwidth (unique weight + BF16 weight-scale bytes, not a hardware DRAM counter)

Each row's effective GB/s = `[N*K + 2*N*(K/SB)] / (median_us * 1000)`; activation, output, scratch, cache misses and duplicate internal traffic are excluded. The comparison NT1 at M16 (baseline) against NT2 at M32 (candidate) tests the >=80% retained-bandwidth gate.

| Shape | Row | NT1 baseline M16 GB/s | NT2 candidate M32 GB/s | Ratio |
|---|---|---:|---:|---:|
| 13312x5120 | iq8r | 736.7 | 695.2 | 94.4% PASS |
| 13312x5120 | i8a8 | 737.7 | 687.5 | 93.2% PASS |
| 5120x6656 | iq8r | 614.6 | 589.3 | 95.9% PASS |
| 5120x6656 | i8a8 | 587.4 | 634.4 | 108.0% PASS |

This is a cold effective-data-rate proxy, **not** direct DRAM measurement. Whether the improvement changes serving performance must be assessed in the separately authorised downstream gate.

## Shim WMMA follow-up candidates (inventory only)

Other active gfx1100 libr3 source sites whose iu8/fp8 WMMA maps through the gfx12 compatibility wrapper. Calls are static source calls per innermost loop iteration; the `j` or `i,j` index means multiplication by the stated number of token/output tiles at runtime. This inventory makes no edits.

| Source (relative to libr4d) | Calls per inner iteration | Across-loop accumulator / possible one-conversion placement |
|---|---|---|
| `r4d_gemm_fp8a8.hip:392` | narrow GEMM: 1 `mma8<I8>` per `ks × NT` | Yes for I8 within one scale group; FP8 path needs numeric proof |
| `r4d_gemm_iqr.hip:453` | IQR narrow: 1 iu8 per `ks × NT` | Yes before its scale fold; **owned by another session** |
| `r4d_gemm_iqr.hip:632` | IQR M64: 1 iu8 per `ks × NT` | Yes before unit fold; **owned by another session** |
| `r4d_moe_iqr.hip:470` | IQR MoE: 1 iu8 per `ks × NT` | Yes per scale group |
| `r4d_gemm_w8a8_nt_m64.hip:62` | W8 narrow: 1 iu8 per helper call | Yes within K segment; preserve fold boundaries |
| `r4d_gemm_w4a8_nt_m64.hip:90` | W4 narrow: 1 iu8 per helper call | Yes, unpacked weight bits; preserve fold |
| `r4d_gemm_w2a8_nt.hip:71` | W2 narrow: 1 iu8 per helper call | Yes, unpacked weight bits |
| `r4d_gemm_w4a8_tiled.hip:49` | W4 tiled: 1 iu8 per helper call | Yes within tile fold |
| `r4d_gemm_w8a8_tiled.hip:170-172` | W8 tiled: 1 conditional iu8 per product | Yes; the first-product zero accumulator must persist |
| `r4d_gemm_wpa8i8_tiled.hip:207-208` | packed-W/A8 tiled: 1 conditional iu8 per `step × (i,j)` | Yes; first-product reset, output layout matter |
| `r4d_gemm_w4a8_prefill.hip:215-217` | W4 prefill: 1 conditional iu8 per K step | Yes; first-product reset and folds |
| `r4d_gemm_8a8_prefill.h:225` | 8A8 prefill FP8 path: 1 `mma8<I8>` per `ks × (i,j)` | Potential FP8; **I8 path at :211 is already native and bypasses shim** |
| `r4d_logits_vocab.hip:290,302` | vocabulary high/low planes: 1 iu8 per `ks × NT` in each of two loops | Yes; convert per plane before its fold |
| `r4d_moe.hip:1443` | MoE main: 1 `mma8<I8>` per K step × `(i,j)` | Yes for I8, honor first-step zero state |
| `r4d_moe.hip:1899` | MoE row: 1 `mma8<I8>` per K step | Yes for I8 |
| `r4d_moe.hip:2044` | MoE narrow: 1 `mma8<I8>` per K step | Yes for I8 |
| `r4d_attn_gen.hip:363-365` | generated attention I8 QK: 2 iu8 per K step | Maybe; softmax boundaries complicate safe conversion |
| `r4d_attn_gen.hip:455` | generated attention FP8 QK/PV: 1 fp8 per inner step | Maybe with BF16-native path; check float order |
| `r4d_attn_gen.hip:483` | generated attention I8: 1 iu8 per inner step | Maybe; preserve attention rescaling |
| `r4d_vsa.hip:687` | VSA I8: 1 iu8 per inner step | Maybe within matrix accumulation |
| `r4d_vsa.hip:828-830` | VSA I8 twin: 2 iu8 per inner step | Maybe; separate accumulators |
| `r4d_vsa.hip:899` | VSA FP8: 1 fp8 per inner step | Maybe; floating-point equivalence required |
| `bc_b16_prefill_impl.h:540` | fused prefill FP8 QK: 1 fp8 per inner `j` | Maybe before attention scale/softmax |
| `bc_b16_prefill_impl.h:636` | fused prefill FP8 PV: 1 fp8 per inner `j` | Maybe before attention rescaling |
| `r4d_gemm_mxfp4a8_nt_m64.hip:166` | MXFP4 narrow: 1 fp8 per `(i,j)` | Maybe per K fold |

Excluded as already native: `r4d_gemm_mxfp4a8_decode.hip`, `r4d_gemm_mxfp4a8_gemv.hip`, `r4d_gemm_mxfp4a8_tiled.hip` have gfx1100 overrides that replace fp8 WMMA calls; `r4d_gemm_8a8_prefill.h` already replaces the I8 branch. The compatibility layer defines an IU4 shim, but there are no direct active IU4 instructions in the compiled source; compressed W2/W4 are expanded to IU8 operands.

**Priority if this experiment passes:** narrow I8 GEMM and vocabulary projection are closest in arithmetic to this change; IQR itself stays with its separate owner. Attention, MOE, and FP8 require separate correctness/precision and profiling gates.

## Executed vs explicitly not executed

**Executed:** native rewrite check; git diff/whitespace review; full 115-unit libr3 build plus r3_selftest build (exit 0, zero stubs); gfx1100 resource metadata for 48 K-parallel and 278 non-K-parallel kernels; full-plugin 2,304/2,304 bit identity across both rows, both dispatch modes, all M1–64 and nine shapes; full-plugin five named r3_selftest cases (1,245 OK, zero FAIL/SKIP); 12 independently leased full-plugin cold replay runs producing 288 records and required bandwidth/latency gates. An additional isolated GEMM-only build and the same bit/cold passes were executed as an early check, but are not substituted for the full plugin.

**Not run:** any serving launch or engine run; ms/step A/B; KLD; end-to-end decoding; multiple GPU card comparison; hardware DRAM counters; full all-rows `r3_selftest` catalogue; cold replay at M49–64 / NT4; push; ship; or release. GPU usage was confined to the explicit short measurement and correctness runs, each with `agent-lease acquire gpu --ttl 12m --est 5m --wait -- <command>`; no GPU lease was held between runs or for builds.

## Reproduction and handoff

Build: `cmake -S engines/radiance/components/libr3 -B <scratch>/build-full -G Ninja -DRADIANCE_SRC=<pinned-radiance> -DCMAKE_PREFIX_PATH=<runtime-prefix> -DR3_TARGETS=gfx1100 -DR3_NATIVE=ON -DBUILD_TESTING=ON`, then `cmake --build <scratch>/build-full --target libr3 r3_selftest -j3` (no `R3_ONLY`). Extract the relevant code-object bundle with `llvm-objdump --offloading <plugin.so>`; inspect metadata with `/opt/rocm-7.2.4/lib/llvm/bin/llvm-readobj --notes <gfx1100-bundle>` (the tested binary was ROCm 7.2.4). Execute the two ABI bitcheck harnesses against baseline/candidate with and without `CHAIN_TEST=1`, then the five `r3_selftest --case` processes. The standalone cold replay uses one lease per baseline/candidate shape arm and its fixed 256 MiB flush / rotated weights / 30 launches; JSONL records and summarising scripts are retained in the experiment scratch directory, outside the repository.

Committed only the branch-local native rewrite as `812f089b9` on `agent/kpar-acc-once`; the worktree is clean. This report remains in the requested brief/report directory. **Not pushed, merged, released, or shipped.**
