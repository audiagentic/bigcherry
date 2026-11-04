# RAD23 experiment: w4-gemv-mr

Archived from branch `perf/w4-gemv-mr` (tip `14f47402ec6dc521b2753dad4b401d3819652595`, base `2c57477bd1f57c07eaf1bc4d84d2ad42d6b886a5`). Not part of any build: the code is kept as an inert patch.
Plan item: RAD23 (see `EXPERIMENT-REGISTER.md` for outcome and reason).

Revisit: `git switch -c exp/w4-gemv-mr <current main> && git apply --3way experiments/radiance/RAD23-w4-gemv-mr/change.patch`.
It may need rebasing: engine files have moved since the base commit.

Commits on the branch:

- 2026-11-01 14f47402e chore(ledger): W4 multi-row GEMV change event
- 2026-11-01 569b61a00 perf(radiance): multi-row packed-f16 W4 decode GEMV (R3_MOE_WORKER_GEMV_W4MR, experimental)
