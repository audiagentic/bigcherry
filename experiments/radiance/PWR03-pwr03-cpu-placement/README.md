# PWR03 experiment: pwr03-cpu-placement

Archived from branch `feat/pwr03-cpu-placement` (tip `c22df881d448be169f5a4518521aba1b6c126192`, base `34d9de39ec1425c53c588206f629202b2cb1bcfb`). Not part of any build: the code is kept as an inert patch.
Plan item: PWR03 (see `EXPERIMENT-REGISTER.md` for outcome and reason).

Revisit: `git switch -c exp/pwr03-cpu-placement <current main> && git apply --3way experiments/radiance/PWR03-pwr03-cpu-placement/change.patch`.
It may need rebasing: engine files have moved since the base commit.

Commits on the branch:

- 2026-10-27 c22df881d docs: resolve merge markers in TOOL_DISPOSITION (keep both sides)
- 2026-10-27 9fbaf3394 lab(radiance): pwr03 idle diagnosis with helpers on efficiency cores
- 2026-10-27 32a67fd71 lab(radiance): pwr03 finds efficiency cores generically (cpu_capacity, else the hybrid PMU); skips when none
- 2026-10-27 6552bf695 lab(radiance): PWR03 helper E-core placement A/B (package power vs tok/s)
