# Recorded heat data

Static expert-heat recordings that lab job files read as inputs.

| File | What it is |
|---|---|
| `perf51.rank0.heat` | Raw per-expert routing counts of engine rank 0 (run perf51, 48 routed layers x 512 experts), the static profile RHP08 compares against |
| `perf51.rank0.hot-first` | The same counts as a hot-first expert ranking, read by `helper-ab.json` through `R3_MOE_WORKER_HEAT` |
| `flashnext-mixed.rank0.heat` | Raw routing counts of engine rank 0 recorded on the Flash-Next four-card layout (`flashnext-4gpu`) under a mixed workload (see below): 586,419,480 routes over 608 requests, 25,003 of the 25,088 expert cells seen |
| `flashnext-mixed.hot-first` | The same counts as a hot-first ranking (`heat-placement/heat-invert.py`), the `--heat` file for the Flash-Next layouts; it replaces the earlier file of this name that was lost from the work root |

Job files reference them as `${BC_REPO_ROOT}/tools/lab/radiance/heat/<file>`. The logs of the recording run are in
`docs/evidence/2026-11-17-work-root-results/runs/perf51/`.

## flashnext-mixed: how it was recorded

- **Layout and build:** `launch-flashnext-best.sh start` defaults (`flashnext-4gpu`: R9700 engine, both XTX cards head, RX 6900 XT tail, helper combine), RAM n-gram placement, main at the release containing the helper VRAM release barrier, `--record-heat` (the engine's route histogram, one-second period), seeded from `perf51.rank0.hot-first`.
- **Workload:** the repository's C++/HIP sources and documentation at equal weight, cut into 900-word chunks (3,386 prompts), 96 greedy completion tokens each, 600 s in four phases of 1, 2, 4 and 8 concurrent sessions: 608 requests, 55,256 completion tokens, no errors.
- **Read it as:** a seed. The live placement policy adapts to the workload actually served; record a new file when the served workload differs from code and documentation. Counts are the engine's cumulative histogram at stop, written by the launcher's stop step.
