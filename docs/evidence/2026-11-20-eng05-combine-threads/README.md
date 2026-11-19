# ENG05: the helper combine thread knee (flashnext-4gpu)

Measured 2026-11-20 05:00-05:35 AEDT (2026-11-19 18:00-18:35 UTC), 20-CPU host, under the `gpu` lease.
Build: main + #1200 (the default is derived; every point sets `helper.combine_threads` explicitly with `--set`).

## Method

`tools/lab/radiance/flashnext-parity.sh measure --launches 1` per point, `FLASHNEXT_PARITY_SETS=helper.combine_threads=N`, with
three changes for this sweep, none of which touch the decode path:

- Placement policy off in every arm. The live-policy calibration store is keyed by a hash of the manifest, and the
  thread count is part of it, so only the stock 8 has a record; a first attempt mixed policy-on (8) and policy-off
  (the rest) arms and was discarded.
- The cold-prefill, document and 256K rounds are skipped (they do not depend on the combine).
- Ten launches in the order 8, 2, 4, 12, 16, 8, 2, 4, 12, 16: two launches per point, with 8 repeated for drift.

Per launch: decode at 1 and 6 streams and 8 sessions (ms/step median), and the helper's own account of its combine
stage (`combine steps N, CPU X ms (Y us a step)`, printed at stop). `runs/pN-tT/` holds the three JSON records and that
line; `runs/table.json` the extracted rows. Draft acceptance was 0.516 in all ten launches.

## Result

| threads | decode1 ms/step (2 launches) | decode6 ms/step | sess8 ms/step | combine stage, us per step |
|---|---|---|---|---|
| 2 | 25.83 / 25.81 | 48.84 / 49.27 | 46.20 / 44.31 | 922 / 926 |
| 4 | 25.81 / 25.74 | 47.88 / 50.05 | 45.78 / 45.69 | 707 / 687 |
| 8 | 25.81 / 25.79 | 48.96 / 48.23 | 45.81 / 45.51 | 588 / 571 |
| 12 | 25.85 / 25.81 | 49.38 / 49.49 | 44.35 / 44.84 | 591 / 591 |
| 16 | 25.83 / 25.87 | 48.86 / 48.50 | 42.50 / 45.71 | 671 / 750 |

- The combine's own cost bottoms out at 8 threads: 922 us at 2, 700 at 4, 580 at 8, 591 at 12, 671-750 at 16.
  The knee is 8; 12 gains nothing and 16 is worse (more threads than the work and the memory bandwidth feed).
- Step time does not move with it. Single-stream decode is 25.74-25.87 ms in all ten launches. Six streams and
  eight sessions vary by about +-1.2 ms between launches of the same point (8: 48.96 and 48.23; 16 on sess8: 42.50 and
  45.71) with no trend in the thread count. Even at 2 threads the extra 340 us per combine step is not visible, so the
  combine is off the step's critical path on this layout and is not part of the v1.17 step-time gap (ENG05 step 1).

## Consequence

`calibration.combine_threads.max = 8` is now a measured seed (evidence in `calibration.toml`). `helper.combine_threads`
resolves as min(8, the CPUs in bc-serve's affinity set) unless a layer sets it.
