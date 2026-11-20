# Archived Radiance lab tools

Retired lab tools are kept here unchanged as evidence of how a measurement was made (same rule as retired patches,
`docs/reference/patches/PATCH_SYSTEM.md`, "Retired patches: archive, never delete"). Nothing under `archive/` is imported,
run or tested by the tooling, the CI or the check gates, and an archived tool is not revived in place: reusing the idea
means a new tool. Each row is one archived directory, moved whole.

| directory | archived | reason | successor | evidence |
|---|---|---|---|---|
| `rsc02_s15/` (`rsc02_s15_ab.py`, `rsc02_s15_projection.py`, `test_rsc02_s15_projection.py`) | 2026-11-20 | RSC02 S15 compared the native owner with the retired Bash launcher (`--legacy-runner`); that launcher no longer exists, so the A/B can no longer be run. Kept as the record of how the native/legacy equivalence was tested. | the native owner's own card-free tests (`bc_session_stored_test`) and the run-record comparison (`bigcherry results compare`) | `docs/reference/engines/radiance/RSC02_S15_ACCEPTANCE.md`, `docs/planning/active/run-serving-configuration/RSC02.md` |
