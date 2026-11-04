# RAD46 experiment: rad46-outputlane

Archived from branch `codex/rad46-outputlane` (tip `b408720c277dac51c88e9ea3141a7dc114127856`, base `7f26008b9a2f6432f9d80304c3b5f6319d39a315`). Not part of any build: the code is kept as an inert patch.
Plan item: RAD46 (see `EXPERIMENT-REGISTER.md` for outcome and reason).

Revisit: `git switch -c exp/rad46-outputlane <current main> && git apply --3way experiments/radiance/RAD46-rad46-outputlane/change.patch`.
It may need rebasing: engine files have moved since the base commit.

Commits on the branch:

- 2026-11-03 b408720c2 wip(radiance): RAD46 outputlane experiment, preserved as left by the Codex session (unvalidated)
- 2026-11-03 34c41ba9a experiment(radiance): preserve grouped rounding in output-lane rows
- 2026-11-03 383c17c1b fix(radiance): retain precision in RDNA2 rows kernel
