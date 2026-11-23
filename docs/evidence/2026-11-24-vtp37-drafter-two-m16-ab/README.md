# VTP37: decode A/B of the drafter two-M16 route, 2020.8.3 against 2020.8.2

Release 2020.8.3 adds one change over 2020.8.2: the DFlash2 drafter's 4-bit IQR linears run M 17-32 as two launches of the spill-free 16-token tile
instead of the spilling 64-token tile (`drafter-two-m16`, see `docs/evidence/2026-11-24-vtp37-drafter-m16/`). That row is not bit-identical to the
stock route (different K order); it was qualified numerically against an FP64 reference. This note records the serving-level measurement.

## Setup

Same harness and layout as `docs/evidence/2026-11-24-vtp37-kpar-acc-once-ab/`: `qwen27b-tp3`, IQ8R linears, DFlash2 drafter, speculation depth 3 in both arms,
untraced, order A1 B1 B2 A2 (A = 2020.8.2, B = 2020.8.3), one launch per arm under its own GPU lease, seeds 1-3, streams 6 and 8, two repetitions,
192 new tokens, temperature 0; three fixed prompts at temperature 0 per arm for a text check.

## Result (within this session only)

| streams | 2020.8.2 ms/step, median (range, n=6) | 2020.8.3 ms/step, median (range, n=6) | change | draft acceptance | tokens/step | aggregate tok/s |
|---|---|---|---|---|---|---|
| 6 | 31.87 (30.06-32.87) | 28.74 (27.97-30.60) | -3.13 ms (-9.8%) | 0.721 / 0.720 | 13.77 / 14.56 | 289.1 / 318.9 |
| 8 | 36.27 (34.57-37.11) | 32.17 (31.54-32.50) | -4.10 ms (-11.3%) | 0.717 / 0.719 | 18.25 / 18.32 | 323.5 / 348.5 |

Arm means over both stream counts: A1 33.74, B1 30.42, B2 30.64, A2 33.89 ms/step. The text of the three fixed prompts is identical in all four arms,
so the route that is not bit-identical produced no visible divergence on these prompts (a small sample, not a quality measurement).

## Reading and caveats

- Draft acceptance is unchanged, so the change is a per-step time effect.
- The 8-stream ranges do not overlap; the 6-stream ranges barely touch.
- Absolute levels drift between sessions: the 2020.8.2 arm at 6 streams was 35.79 ms in the earlier 8.1-against-8.2 session and 31.87 ms here.
  Compare only within a session. The two deltas (kpar alone, drafter alone) are each within-session and must not be added or chained into a cumulative figure.
- Serving quality (KLD) of the new drafter route is not measured here.
