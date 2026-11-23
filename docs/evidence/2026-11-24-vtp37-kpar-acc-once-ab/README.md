# VTP37: decode A/B of the K-parallel accumulate-once change, 2020.8.2 against 2020.8.1

Release 2020.8.2 contains one change over 2020.8.1: the K-parallel int8 tile keeps its WMMA sums in the gfx11 layout per scale group and converts once
(`kpar-acc-once`, see `docs/evidence/2026-11-24-vtp37-kpar-acc-once/`). This note records the serving-level measurement.

## Setup

- Layout `qwen27b-tp3` (ranks 0 and 1 on RX 7900 XTX, rank 2 on R9700), IQ8R linears, DFlash2 drafter, speculation depth 3 (same stored configuration in both arms).
- Installed release roots, same machine, same weights, power profile stock, untraced (no step-phase trace).
- Order A1 B1 B2 A2 (A = 2020.8.1, B = 2020.8.2), one engine launch per arm, each arm under its own GPU lease.
- Per arm: seeds 1-3, streams 6 and 8, two repetitions each, 512-token prompts target (271 words), 192 new tokens, temperature 0.
  Reported: engine `decode_ms_per_step`, `decode_tokens_per_step`, draft acceptance, aggregate tokens per second.
- Text check: three fixed prompts at temperature 0, 96 tokens, per arm.

## Result

| streams | 2020.8.1 ms/step, median (range, n=6) | 2020.8.2 ms/step, median (range, n=6) | change | draft acceptance 8.1 / 8.2 | tokens/step 8.1 / 8.2 | aggregate tok/s 8.1 / 8.2 |
|---|---|---|---|---|---|---|
| 6 | 39.37 (36.40-43.62) | 35.79 (33.71-39.82) | -3.58 ms (-9.1%) | 0.725 / 0.732 | 16.41 / 16.73 | 287.2 / 309.3 |
| 8 | 39.06 (37.80-40.27) | 36.44 (35.21-38.27) | -2.62 ms (-6.7%) | 0.720 / 0.712 | 18.36 / 18.13 | 316.6 / 327.8 |

Arm means over both stream counts: A1 38.62, B1 36.06, B2 36.65, A2 39.76 ms/step (both 2020.8.2 arms below both 2020.8.1 arms).
The three fixed prompts produced identical text in all four arms.

## Reading

- The gain agrees with the replay prediction (nt 2 GEMMs -28% in isolation, about 3.5 ms of the step expected; measured 2.6-3.6 ms).
- Draft acceptance and tokens per step do not move, so the change is a per-step time effect, not a different draft behaviour.
- Caveat: the spread between seeds is about 3 ms. The 6-stream ranges of the two releases overlap; the 8-stream ranges nearly separate.
  Treat the size as approximate and the direction as supported by both arm orders and the replay.

## Harness note

The 2020.8.1 install used here had extra files in its release root (code objects extracted from the plugin for an earlier static analysis).
`bc-serve` refused to start with `BC_RELEASE_6: runtime file membership changed`, which is the intended check. This is a fact about the harness, not a
product defect. The extra files were removed after confirming every one was absent from the bundle file list, and the arm was rerun.
