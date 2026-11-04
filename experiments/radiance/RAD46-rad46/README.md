# RAD46 experiment: rad46

Archived from branch `codex/rad46` (tip `ffac8c81f5f31f951e50d0cc2afceb4058e7dbb8`, base `7f26008b9a2f6432f9d80304c3b5f6319d39a315`). Not part of any build: the code is kept as an inert patch.
Plan item: RAD46 (see `EXPERIMENT-REGISTER.md` for outcome and reason).

Revisit: `git switch -c exp/rad46 <current main> && git apply --3way experiments/radiance/RAD46-rad46/change.patch`.
It may need rebasing: engine files have moved since the base commit.

Commits on the branch:

- 2026-11-03 ffac8c81f docs(radiance): record RAD46 kernel profiling
- 2026-11-03 0e96deae7 docs(radiance): record RAD46 prefill regression
- 2026-11-03 5ff90b027 docs(radiance): record deterministic RAD46 quality captures
- 2026-11-03 33ead25a9 docs(radiance): record RAD46 outlier and candidate evidence
- 2026-11-03 ba8d10604 docs(radiance): record RAD46 stage comparisons
- 2026-11-03 c7758e726 docs(radiance): record RAD46 quality diagnostics
- 2026-11-03 383c17c1b fix(radiance): retain precision in RDNA2 rows kernel
