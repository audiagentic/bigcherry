---
id: MEN16
order: 0
plan: run-multi-engine
state: pending
created-at: '2026-10-17T09:09:28.078742+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P1
work: M
---

# pin-bump builds its own lane and runs the CPU smoke on it

## Description

Owner-approved (2026-10-17) automation from the first radiance bump: today the bump is build lane -> pass the build dir to pin-bump --cpu-smoke-build -> pin-bump. pin-bump should build the bumped engine itself through the canonical lane (source from --source, build profile, platform from --platform) after rebase/apply, record the build identity in the release record, and run the cpu-smoke phase against that build. --cpu-smoke-build stays only as an explicit override recorded in the resume state.

## Steps

1. Add a 'lane-build' phase before cpu-smoke using campaign build machinery (no reimplementation). 2. Record build_plan_id and binary hashes in the release record. 3. cpu-smoke consumes it. 4. Tests with fake build runner; docs PIN_BUMP.md.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

A radiance pin-bump run with only --source and --platform completes on dev; offline tests for phase order and resume.

## Effort & Risk



## Standards



## Acceptance Criteria

**Transferred MEN22 acceptance (independent gate):** MEN16 lane-build must commit only its own release-record paths (releases/index.json and the exact release file for target, never unrelated staged edits or the pin-transition marker) BEFORE the controller's clean-tree lane build, using existing _commit_release_records in tools/bigcherry/release/pin_bump.py. Assert repeat/resume idempotence and both Radiance and llama.cpp phase order; reject blanket allow-dirty. Preserve MEN22's historical v1.8.0 LANE_BUILD_FAILED and the 5/5 offline exact-path tests, but require one source-linked real subsequent first-run or resumed pin-bump receipt proving no manual release-record commit intervened, with run ID, source ref, release JSON, git commit history, passed lane-build and CPU smoke. Confirm PIN_BUMP.md documents record-before-lane behavior and complete/reaudit. Do not close MEN16 on tests or v1.17.0 release record alone; it still needs a source+platform-only Radiance bump which builds one canonical lane, records build_plan_id/binary hash and passes CPU smoke without manually provided --cpu-smoke-build. No second bump driver or release-record writer.

## Notes

CONSOLIDATION 2026-11-22 PHASE-5: receive MEN22. MEN22 is a failure mode in MEN16's sole pin-bump lane-build phase; the shipped pre-lane _commit_release_records belongs to the same orchestrator and should have one acceptance owner. Original acceptance still applies; the transferred gate is unfinished. Source is superseded for duplicate planning ownership, not completed on implementation, benchmark or live-hardware criteria. Where Vulkan is involved, implementation remains paused by owner directive.

## Change Log

- 2026-10-17T09:09:28.078742+00:00 (created-by): Created by agent

## Reviews

- RV4527
- 2026-11-22T11:08:17.000227+00:00 (updated-by): Updated: section:acceptance_criteria, section:notes
