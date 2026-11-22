---
id: MEN22
order: 22
plan: run-multi-engine
state: superseded
created-at: '2026-10-18T13:53:54.146144+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P1
work: S
---

# pin-bump lane-build refuses the dirty checkout its own release record leaves

## Description

First real radiance bump (v1.7.0 -> v1.8.0, 2026-10-19): after drift, compile and the component gates passed, the lane-build phase (MEN16) stopped with LANE_BUILD_FAILED: 'BigCherry repository is dirty; use explicit development override'. The dirty files were releases/index.json and releases/v1.8.0.json, written by the orchestrator itself in earlier phases. The orchestrator must leave a clean tree for the phases that need one: commit the release record (in its own commit, like the repin) before lane-build, or have the lane build ignore exactly the orchestrator's release-record paths. Do not pass a blanket allow-dirty.

## Steps

1. Verify `pin_bump.py` calls `_commit_release_records` before `run_phase_lane_build`, and that exact release paths only are committed, not other staged changes or the pin-transition marker; check both engine selections.
2. Run `tools/tests/release/test_pin_bump.py` release-record and lane-order tests, including fake dirty tree, idempotent rerun, selective commit, resume and both-engine path.
3. Identify or capture one later real `pin-bump` first-run/resume receipt showing that lane-build saw a clean controller checkout without a manual release-record commit; include run ID, engine, ref, release record and Git evidence.
4. Verify `docs/reference/build/PIN_BUMP.md` documents the pre-lane record commit and subsequent complete/reaudit behavior before closure.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Five `CommitReleaseRecordsTests` passed on 2026-11-22. A subsequent real bump receipt with no manual release record commit is still required; `releases/v1.17.0.json` records a passed lane build but alone is not conclusive for that negative assertion.

## Effort & Risk



## Standards



## Acceptance Criteria

1. The orchestrator commits exactly its own release records before lane-build, preserves unrelated dirty paths/staged content, is idempotent on resume and continues updating records on complete/reaudit.
2. Offline tests cover exact-path commit and both Radiance/llama.cpp phase ordering, plus clean-tree lane admission under injected prior record writes; no blanket allow-dirty.
3. One real bumped-engine lane build succeeds on the first or resumed orchestrator path, with a verifiable record that no manual release-record commit was needed.
4. The workflow is documented under `docs/reference/build/PIN_BUMP.md`. Do not mark complete solely on passing unit tests or a release record without the provenance in criterion 3.

## Notes

Worked around on the bump branch by committing the record by hand. Found by the radiance 1.8.0 bump.

CONSOLIDATION 2026-11-22: source code `tools/bigcherry/release/pin_bump.py` commits record paths before lane build; 5/5 focused exact-path tests pass. A later v1.17.0 record shows lane build and CPU smoke succeeded, but provenance of no manual intervention is not independently established here; the real bump criterion remains open.

CONSOLIDATION 2026-11-22 PHASE-5: MEN22 is a duplicate planning owner of MEN16. MEN22 is a failure mode in MEN16's sole pin-bump lane-build phase; the shipped pre-lane _commit_release_records belongs to the same orchestrator and should have one acceptance owner. The exact independently testable acceptance and negative controls now live in active MEN16. Retire this planning owner without changing any existing implementation, hardware validation claim or original source evidence. All original design and measured failure records remain in this archived source document.

## Change Log

- 2026-10-18T13:53:54.146144+00:00 (created-by): Created by agent

## Ledger-events

- chg_20261018_140640_radiance-is-now-on-upstream-1_2179
- 2026-10-18T14:06:48.708564+00:00 (updated-by): Updated: section:ledger-events

## Reviews

- RV4473
- 2026-11-22T09:20:16.188520+00:00 (updated-by): Updated: section:steps, section:validation, section:acceptance_criteria, section:notes
- RV4521
- 2026-11-22T11:09:49.800809+00:00 (updated-by): Updated: section:notes
- 2026-11-22T11:13:49.269664+00:00 (state-transition): State: pending → superseded
