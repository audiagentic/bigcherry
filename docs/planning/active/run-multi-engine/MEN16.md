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

Wire RCS19 existing composition gate at exact materialization/coverage/apply boundary, before expensive compile (pristine coupling remains preflight). Use existing compile-tree and lane build checks, recorded git commits, ordered patch selection and recipe, origin/integrated revisions, component versions, SDK/compiler/arch/options, gate versions and results. Compare plain fields in existing state.json/receipts; re-run affected and later dependent gates on change or uncertainty. Preserve existing build hashes only; no new composite digest, overlay digest or proof-key system. Config/profile/effective power and qualification identities bind runtime/cache/acceptance, not indiscriminately compile key. Compile-to-lane reuse only identical inputs/options/targets and artifacts; no duplicate build path. Pass immutable candidate lane/bundle identity to MEN17. Explicit build override never silently used.

## Code Samples & Guidance



## Files



## Validation

A radiance pin-bump run with only --source and --platform completes on dev; offline tests for phase order and resume.

## Effort & Risk



## Standards



## Acceptance Criteria

**Transferred MEN22 acceptance (independent gate):** MEN16 lane-build must commit only its own release-record paths (releases/index.json and the exact release file for target, never unrelated staged edits or the pin-transition marker) BEFORE the controller's clean-tree lane build, using existing _commit_release_records in tools/bigcherry/release/pin_bump.py. Assert repeat/resume idempotence and both Radiance and llama.cpp phase order; reject blanket allow-dirty. Preserve MEN22's historical v1.8.0 LANE_BUILD_FAILED and the 5/5 offline exact-path tests, but require one source-linked real subsequent first-run or resumed pin-bump receipt proving no manual release-record commit intervened, with run ID, source ref, release JSON, git commit history, passed lane-build and CPU smoke. Confirm PIN_BUMP.md documents record-before-lane behavior and complete/reaudit. Do not close MEN16 on tests or v1.17.0 release record alone; it still needs a source+platform-only Radiance bump which builds one canonical lane, records build_plan_id/binary hash and passes CPU smoke without manually provided --cpu-smoke-build. No second bump driver or release-record writer.

Tests in test_pin_bump.py, test_pin_bump_resume_inputs.py and engine adapter fixtures: exact phase order for Radiance/llama.cpp, interrupted composition/compile/lane/smoke/accept, changed patch/header/origin/gate/schema/config/lane/binary invalidates correct proof, wrong target or required skipped/unverified stops, valid unchanged resume reuses. Real source+platform-only candidate bump with own release records, actual canonical lane and CPU smoke; required hardware evidence attached before qualification, no manual cursor/record repair. Existing MEN22 acceptance unchanged.

## Notes

CONSOLIDATION 2026-11-22 PHASE-5: receive MEN22. MEN22 is a failure mode in MEN16's sole pin-bump lane-build phase; the shipped pre-lane _commit_release_records belongs to the same orchestrator and should have one acceptance owner. Original acceptance still applies; the transferred gate is unfinished. Source is superseded for duplicate planning ownership, not completed on implementation, benchmark or live-hardware criteria. Where Vulkan is involved, implementation remains paused by owner directive.

2026-11-25 cross-programme bump review: docs/design/radiance-bump-integration.md is the shared implementation-order and compatibility specification; MEN24 owns sequence and timing, domain implementation remains here. No duplicate checker, bump driver, manifest launch or legacy fallback. Existing run_phase_drift on pristine source and _compile_tree_oid vendor+component identity verified. Preserve existing selector freeze, exact-path release record commit before lane, transition semantics and canonical build. RCS19 owns composition checker, MEN17 accepts candidate. Use one engine-generic phase API via adapters; no blanket dirty bypass.

## Change Log

- 2026-10-17T09:09:28.078742+00:00 (created-by): Created by agent

## Reviews

- RV4527
- 2026-11-22T11:08:17.000227+00:00 (updated-by): Updated: section:acceptance_criteria, section:notes
- RV4604
- 2026-11-25T02:03:42.673255+00:00 (updated-by): Updated: section:notes, section:detailed_solution, section:acceptance_criteria
