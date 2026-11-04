# RAD46 experiment: rad46-attribution

Archived from branch `codex/rad46-attribution` (tip `a5c07d1ea3cbbe4d2474af2cdd052274e518fdc0`, base `7f26008b9a2f6432f9d80304c3b5f6319d39a315`). Not part of any build: the code is kept as an inert patch.
Plan item: RAD46 (see `EXPERIMENT-REGISTER.md` for outcome and reason).

Revisit: `git switch -c exp/rad46-attribution <current main> && git apply --3way experiments/radiance/RAD46-rad46-attribution/change.patch`.
It may need rebasing: engine files have moved since the base commit.

Commits on the branch:

- 2026-11-03 a5c07d1ea audit(radiance): log route tickets from helper
- 2026-11-03 ae46db9a8 audit(radiance): trace RAD46 diagnostic routes
- 2026-11-03 383c17c1b fix(radiance): retain precision in RDNA2 rows kernel
