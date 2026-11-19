# Flash-Next four-GPU regression against the v1.17.0 baseline (2026-11-19)

Question: the Flash-Next four-GPU parity baseline (v1.17.0, 2026-11-07) is faster than a current main build. Which
differences are real, and what do they say about how runs are recorded?

All runs: one same-session pass on the Flash-Next four-GPU layout, `tools/lab/radiance/flashnext_parity.py` through
`flashnext-parity.sh`, one launch per arm. Run folders are in the work root (`<work-root>`, scratch
`tmp/bigcherry/regress-ab/`) and are not copied here. Times are AEDT; the driver log is `driver.log` in that folder.

## Outcome (end of 2026-11-19)

Main with all of the day's fixes (#1162, #1168, #1169, #1174, #1176, #1177, #1178), live placement policy on, flashnext-mixed
heat, 451 s policy warmup (`FLASHNEXT_PARITY_WARMUP_S=451`, the v1.17.0 baseline's warmup), two launches, compared with
the v1.17.0 baseline (`docs/reference/engines/radiance/baselines/flashnext-4gpu-parity.json`):

| metric | v1.17.0 | main, policy on (launches) | change |
|---|---|---|---|
| 256K prefill long7 / long8 tok/s | 2914 / 2901 | 3085 (3074, 3096) / 3183 (3167, 3199) | +5.9 % / +9.7 % |
| 44K document prefill tok/s | 3591 | 3742 (3741, 3743) | +4.2 % |
| cold 27K TTFT s | 7.95 | 8.30 (8.27, 8.32) | +4.3 % |
| 8-session TTFT s | 20.93 | 21.29 (21.34, 21.24) | +1.7 % |
| decode 1 stream ms/step | 24.99 | 25.87 (25.86, 25.89) | +3.5 % |
| decode 6 streams ms/step | 47.04 | 48.68 (47.80, 49.55) | +3.5 % |
| 8-session ms/step | 28.89 | 30.20 (30.39, 30.01) | +4.5 % |
| 256K-context decode ms/step | 27.82 | 28.24 | +1.5 % (within allowance) |
| policy commits | 7 | 5, 5 | |

Prefill is now faster than v1.17.0 at every length; TTFT is within 2-4 %. Decode tok/s equals v1.17.0 (1 stream ~113 vs
~114, 6 streams ~376 vs ~377 by (1 + 3 x acceptance) / step) while each step is ~3.5 % slower; that gap is planned in
tuning-radiance-engine/ENG05 (about 2 % arrived after 2026-11-09: release 1170.14.0 25.3 ms/step vs main 25.9, same session,
policy off). Policy off, the same main is 10.1 s cold / 27.3 s sess8 / 2419-2464 tok/s at 256K (two launches).

What closed the gap:
1. **Heat file** (#1168): the parity harness defaulted to `perf51.rank0.hot-first`, not the launcher's
   `flashnext-mixed.hot-first`; perf51 cost ~10 % of 256K prefill (2216/2185 vs 2419/2464 tok/s, policy off).
2. **Live policy back on** after five defects on a path that had never run natively: calibration refused
   unwatermarked heat silently and ran ~67 h (#1162); calibration and production computed different keys and the key
   moved with derived budgets/host pool (#1169); calibration prompts A (repository docs) and B (repository code) routed
   too similarly for a drift half-life, while long synthetic filler vs ~44K-token documents converges (the Nov 10
   method); the heat guard refused a valid empty first snapshot and printed only "2" (#1174); the runtime receipt
   refused the fresh-boot placement state (#1176); a second start refused its own stored manifest (#1177).
3. **Warmup**: the policy derived a 71 s warmup (min_samples 2 x cadence 23 s); with it cold TTFT was 8.75 s; 451 s gave
   8.06-8.30 s.

Ruled out on the way: code since 2026-11-09 for prefill (r14 as slow as main, policy off), helper budgets (old
22528/22528/14336 MiB: no change), XTX power (2.9-3.0 GHz, 150-210 W under 256K; `power1_cap=0` is a kernel getter
defect), R9700 PCIe x4 (unchanged since before the baseline), prompt content (long and cold prompts are seeded synthetic
words, unchanged since #721).

Build tooling found on the way: content-addressed build trees collided across checkouts (#1172); random compiler
crashes at ~1/160 heavy compiles in three different programs with ample memory and no ccache involvement, pointing at
host RAM/CPU stability (host checks requested; one recorded retry per crashed build, #1175). Reproduce by taking a libr3
unit's command from `compile_commands.json` of a libr3 component build (e.g. `units/r4d_vit_bf16.hip`) and running it
~160 times, counting signal deaths.

## Symptoms

| Metric | v1.17.0 baseline | main (738a48d6e) | r14 (bc-radiance-1170.14.0, 2026-11-09) | main, old helper budgets |
|---|---|---|---|---|
| cold 27K TTFT (s) | 7.95 | 11.90 | 10.60 | 11.84 |
| doc prefill (tok/s) | 3591 | 3706 | 3709 | 3695 |
| long7 prefill (tok/s) | 2914 | 2216 | 2330 | 2212 |
| long8 prefill (tok/s) | 2901 | 2185 | 2290 | 2210 |
| 8-session TTFT (s) | 20.93 | 32.07 | 28.89 | 31.66 |
| decode1 (ms/step) | 24.99 | 25.73 | 25.28 | 25.67 |
| decode6 (ms/step) | 47.04 | 47.55 | 49.32 | 47.71 |

Baseline values are the stored `docs/reference/engines/radiance/baselines/flashnext-4gpu-parity.json`; the other
columns are `<arm>/parity.json` medians (`main`, `r14`, `main-oldbudget`). "Old helper budgets" are 22528 / 22528 /
14336 MiB (main today derives 23074 / 23074 / 14882). A main-arm parity file was also stored as
`baselines/flashnext-4gpu-parity-main-20261119.json`.

Acceptance rates differ between arms (decode1 0.80 on main, 0.61 on r14 and the baseline) with a single launch each,
so step times at these rates are not a clean decode comparison. Decode is within about 3 % everywhere; the
regressions are cold TTFT, 8-session TTFT and long-context prefill.

## Cause 1, confirmed: the live placement policy has been off since 2026-11-09

The baseline recorded 7 policy transactions; main records 0. The calibration key changed in #790, so the stored
calibration record no longer matched and the policy never started. Same-build ABBA (RHP16): cold 27K 7.43 to 7.46 s
with the policy on, 10.86 to 10.95 s off; 8-session TTFT about 23 s on, about 34 s off. This explains the cold and
8-session deficits (about 32 % on cold). It does not explain 256K prefill.

## Cause 2, open: 256K prefill about 20 % below baseline, even on 2026-11-09 code

Ruled out by measurement:

- Code since 2026-11-09: 5 to 12 % at most (r14 vs main above).
- Helper budgets: no change with the old budgets (last column).
- XTX power: 2.9 to 3.0 GHz and 150 to 210 W under 256K load (`power-oldbudget.txt`). The navi31 `power1_cap` file
  reads 0 on kernel 7.0.13, which is a getter defect, not a cap of zero.

Not ruled out:

- Heat file. The baseline's `flashnext-mixed.hot-first` is lost; the re-recorded file differs.
- Host kernel `7.0.13-gfxoff` (built 2026-11-11). The BHK01 step 7 serving check was never done; one results-store record
  from that kernel on 2026-11-11 still had cold 27K at 7.53 s, so the kernel does not explain cold TTFT.
- Host pool: baseline launcher fixed 9216 MiB, main measures 2247 MiB.

## Calibration failure, root cause

The calibration input policy had `accept_unwatermarked_heat=false`. Heat read from a file is never watermarked, so every
heat delta was rejected and the no-feed bound was about 67 hours. #1162 fixed or refused this; a rerun with the
2026-11-10 derived policy converged (accepted heat deltas above 100).

## What the investigation could not tell without guessing

- The baseline has no manifest and no heat file: what it ran is unrecoverable, so "same configuration" is unprovable.
- Nothing said the policy was off until a transaction count was compared by hand, nine days later.
- Results records stored a manifest path, and those manifests were deleted in the 2026-11-17 purge.
- The measured arms needed host facts (kernel, clocks, power cap) collected by hand.

These are the motivation for plan `run-measurement-records` (MRC01 to MRC06): a content-addressed run record with a
provenance bundle, host facts, feature states, a calibration store keyed only by what changes the calibration, and
baseline/compare/history tools over records.
