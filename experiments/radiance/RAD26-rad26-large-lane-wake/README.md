# RAD26 experiment: rad26-large-lane-wake

Archived from branch `feat/rad26-large-lane-wake` (tip `eb439f245a7bfaa3b278684e3604959c7292fad5`, base `72beabbc2357d1185f90122983b816e5ee96a800`). Not part of any build: the code is kept as an inert patch.
Plan item: RAD26 (see `EXPERIMENT-REGISTER.md` for outcome and reason).

Revisit: `git switch -c exp/rad26-large-lane-wake <current main> && git apply --3way experiments/radiance/RAD26-rad26-large-lane-wake/change.patch`.
It may need rebasing: engine files have moved since the base commit.

Commits on the branch:

- 2026-10-27 eb439f245 diag(radiance): RAD26 first-wake log and rad26-diag.sh
- 2026-10-27 c2506afbf diag(radiance): RAD26 wake counters - engine wakes queued/captured/fired, helper large-lane notify wakes
- 2026-10-27 f284aa420 lab(radiance): rad26 graph mode - publish + wake callback captured in a HIP graph, replayed; reports any replay whose
- 2026-10-27 dc724c46c feat(radiance): engine registration timeout is a setting (R3_MOE_WORKER_REGISTER_TIMEOUT_S, default 120)
- 2026-10-27 b48e46e62 feat(radiance): large lane sleeps until a host callback after the engine's publish wakes it (RAD26)
