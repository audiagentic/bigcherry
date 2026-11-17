# Recorded heat data

Static expert-heat recordings that lab job files read as inputs.

| File | What it is |
|---|---|
| `perf51.rank0.heat` | Raw per-expert routing counts of engine rank 0 (run perf51, 48 routed layers x 512 experts), the static profile RHP08 compares against |
| `perf51.rank0.hot-first` | The same counts as a hot-first expert ranking, read by `helper-ab.json` through `R3_MOE_WORKER_HEAT` |

Job files reference them as `${BC_REPO_ROOT}/tools/lab/radiance/heat/<file>`. The logs of the recording run are in
`docs/evidence/2026-11-17-work-root-results/runs/perf51/`.
