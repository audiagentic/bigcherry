# RSC02 step 6 — progress and remaining gates

Updated: 2026-11-18 (card-free work only; no hardware lease).

## Merged on main (verified in RSC04 and the component)

- #1078: native `bc-serve explain CONFIG`.
- #1081: typed native advice from the heat watcher.
- #1082: typed named-config lab settings and native resolver/writer tests.
- #1080: S15 card-free owner-parity rehearsal and semantic projection.
- RSP01 #1077, #1079, #1086: release work owned by another lane; do not modify it here.

## Milestone: S15 process-capture role binding (this PR)

- RED test first: `test_refuses_matching_but_incomplete_observed_roles` exercises two *equally incomplete* snapshots, a phantom role, a forged `--role` and an omitted `--role`. Before the production change, agreement between two snapshots could be reported as PASS even though the same helper was absent from both.
- Production comparison now obtains the expected role set from the captured manifest `spec.processes.helpers[].role`, requires every role to have one matching `--role NAME` argument, and fails on missing/extra observed roles. This is an independent admission condition, not a replacement for the real manifest writer/strict reader or the fake-process owner rehearsal.
- The original S15 dry-run and real owner CTest are left intact.
- Validation status: GitHub CI pending when this note was written. The intended local worktree `/work/development/tmp/bigcherry/rsc02-step6-lab-experiment` was unavailable in this execution environment, so no local CTest, ASan/UBSan, clang-tidy, clang-format, full Python suite, patch-lint or forbidden-terms result is claimed.

## Remaining step-6 work (separate PRs)

1. Recover, rebase, validate and publish the two local `feat/rsc02-step6-lab-experiment` commits; the branch is not visible on the remote. Integrate lab calibration and typed experiment modes through the production config resolver, writer and session owner (RSC12 dependencies documented in RSC04).
2. Add explicit typed profiling/child-wrapper capability and card-free production-path tests; never forward `HELPER_WRAP`, `HELPER0_WRAP` or arbitrary inherited `R3_*` as configuration.
3. Migrate the named lab launchers and their behavioural test assertions to native commands and compiled CTest/card-free CLI tests. Do not remove the old launchers until all callers are proven.
4. Complete the leased S15 harness lifecycle adapter and same-build dense/helper runbook gates (including four clean ABBA pairs per cohort). The existing `docs/reference/engines/radiance/RSC02_S15_ACCEPTANCE.md` specifies evidence and blocking S14/RHP12 dependencies; the card-free rehearsal does not provide GPU acceptance.
5. Create the separate **DO NOT MERGE** S16 deletion PR only after caller-migration proofs, keeping it unmerged until leased S15 and RSP01 release acceptance pass. Audit runtime, test and indirect callers with `git grep` before deletion.

No GPUs were used and no files in `tools/bigcherry/release` or the release workflows were modified.
