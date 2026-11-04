# RAD46 experiment: rad46-quadadd

Archived from branch `codex/rad46-quadadd` (tip `f049dcc5c9438736eab927dff0e06636fa88cd40`, base `7f26008b9a2f6432f9d80304c3b5f6319d39a315`). Not part of any build: the code is kept as an inert patch.
Plan item: RAD46 (see `EXPERIMENT-REGISTER.md` for outcome and reason).

Revisit: `git switch -c exp/rad46-quadadd <current main> && git apply --3way experiments/radiance/RAD46-rad46-quadadd/change.patch`.
It may need rebasing: engine files have moved since the base commit.

Commits on the branch:

- 2026-11-03 f049dcc5c wip(radiance): RAD46 quadadd experiment, preserved as left by the Codex session (unvalidated)
- 2026-11-03 34c41ba9a experiment(radiance): preserve grouped rounding in output-lane rows
- 2026-11-03 383c17c1b fix(radiance): retain precision in RDNA2 rows kernel
