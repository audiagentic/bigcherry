---
id: PCC01
order: 0
plan: patching-code-cleanup
state: completed
created-at: '2026-09-09T10:47:19.397245+00:00'
breadth: ''
skill: intermediate
created-by: capability-rebaseline-v3
work: S
priority: P2
---

# Fix stale tests for patches 1225 (HI85), 1233 (RD73), 1242 (HI134): implementation drifted past its own tests

## Description

Patch-test regressions had drifted from both patch manifests and the capability-rebaseline plan namespace. RD73's test omitted its VA06 common.cuh target, HI134's test omitted declared 0830 dependency application, and the NRO package-shape test still expected retired nasone-rdna plan paths/IDs. Production patch manifests and selection state remain unchanged.

## Steps

1. Implement the still-valid future scope.
2. Run the stated acceptance and evidence gates.
3. Preserve predecessor provenance and record successor evidence under this ID.

## Detailed Solution & Technical Design

Capability owner: patching.

Keep the production patch manifests unchanged. Make each test fixture represent the package contract: copy every target file touched by the patch set, and apply declared dependencies before the dependent patch. This preserves real pinned-source and idempotence/composition coverage while preventing tests from passing against an incomplete synthetic tree.

Split assessment: one independent boundary; Build/Run support is a dependency. Overlap assessment: no duplicate boundary found; related items are prerequisites or adjacent evidence.

## Code Samples & Guidance



## Files

- successor-specs/patching-code-cleanup-co02.md
- tools/tests/patch/test_patch_1233_rd73_graph_cache_key.py
- tools/tests/patch/test_hi134_meta_stage_trace.py
- tools/tests/patch/test_nro_patch_packages.py

## Validation

Run pytest tools/tests/patch -q. The focused fixed tests pass (RD73/HI134: 9 passed; NRO package shape: 13 passed; HI85 guard: 4 passed). The full patch suite must pass except for explicitly skipped tests.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

All stale tests pass against the current real pinned vendor source and current PNRO successor plan namespace; idempotence, dependency application, and composition coverage remain enforced. No production patch implementation or selection state changes are made.

## Notes

Supersedes: CO02
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-code-cleanup-co02

2026-09-10: Fixed PCC01 test drift. RD73 fixtures now include common.cuh for the VA06 edit; HI134 fixtures apply 0830 before 1242. Focused suite: 9 passed; HI85 guard: 4 passed.

2026-09-10: Fixed PCC01 test drift. RD73 fixtures now include common.cuh for the VA06 edit; HI134 fixtures apply 0830 before 1242; NRO package-shape tests now validate PNRO01..PNRO15 while retaining historical NRO manifest IDs for provenance.

## Change Log

- 2026-09-09T10:47:19.397245+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:03:20.832031+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events



- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:00.735328+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-09T15:10:37.489854+00:00 (state-transition): State: pending → in_progress
- 2026-09-09T15:11:05.529654+00:00 (updated-by): Updated: section:description, section:detailed_solution, section:files, section:validation, section:acceptance_criteria, section:notes
- 2026-09-09T15:11:25.711089+00:00 (updated-by): Updated: section:detailed_solution, section:files, section:validation
- chg_20260909_151141_fixed-stale-patch-tests-so-rd7_2014
- 2026-09-09T15:11:41.022339+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-09T15:15:00.454880+00:00 (updated-by): Updated: section:description, section:files, section:validation, section:acceptance_criteria, section:notes
- chg_20260909_151742_aligned-the-remaining-package_2235
- 2026-09-09T15:17:43.006339+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-09T15:18:01.335366+00:00 (state-transition): State: in_progress → completed
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.175993+00:00 (updated-by): Updated: section:ledger-events
