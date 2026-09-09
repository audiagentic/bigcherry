---
id: PCC01
order: 0
plan: patching-code-cleanup
state: pending
created-at: '2026-09-09T10:47:19.397245+00:00'
breadth: ''
skill: intermediate
created-by: capability-rebaseline-v3
work: S
priority: P2
---

# Fix stale tests for patches 1225 (HI85), 1233 (RD73), 1242 (HI134): implementation drifted past its own tests

## Description

Patch-test regressions have partial lineage fixes, but CO02 still owns unresolved cleanup including the HI104 nested-runtime failure.

## Steps

1. Implement the still-valid future scope.
2. Run the stated acceptance and evidence gates.
3. Preserve predecessor provenance and record successor evidence under this ID.

## Detailed Solution & Technical Design

Capability owner: patching

Split assessment: One independent boundary; Build/Run support is a dependency.

Overlap assessment: No duplicate boundary found; related items are prerequisites or adjacent evidence.

## Code Samples & Guidance



## Files

successor-specs/patching-code-cleanup-co02.md

## Validation

Historical evidence and constraints: Preserve frozen notes/reviews/evidence on predecessor; IDs: CO01,HI134,HI85,RD73,VA25.

Active dependencies: Frozen dependencies: none recorded.

Reference handling: Rewrite forward references (2); preserve historical references (5) on predecessor.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Close the recorded gap and pass the frozen scope's stated implementation, correctness, performance, or evidence gate.

## Notes

Supersedes: CO02
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-code-cleanup-co02

## Change Log

- 2026-09-09T10:47:19.397245+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:03:20.832031+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:00.735328+00:00 (updated-by): Updated: section:ledger-events
