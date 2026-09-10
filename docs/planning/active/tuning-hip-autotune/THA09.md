---
id: THA09
order: 0
plan: tuning-hip-autotune
state: pending
created-at: '2026-09-09T10:49:14.603876+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: P1
---

# Lazy native_select on the dispatch hot path (reopens HI87 under the sub-1% policy)

## Description

Lazy-native-select optimization remains planned and unimplemented.

## Steps

1. Implement the still-valid future scope.
2. Run the stated acceptance and evidence gates.
3. Preserve predecessor provenance and record successor evidence under this ID.

## Detailed Solution & Technical Design

Capability owner: tuning

Split assessment: One independent boundary; Build/Run support is a dependency.

Overlap assessment: No duplicate boundary found; related items are prerequisites or adjacent evidence.

## Code Samples & Guidance



## Files

successor-specs/tuning-hip-autotune-hi158.md

## Validation

Historical evidence and constraints: Preserve frozen notes/reviews/evidence on predecessor; IDs: HI87.

Active dependencies: Frozen dependencies: none recorded.

Reference handling: Rewrite forward references (1); preserve historical references (1) on predecessor.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Close the recorded gap and pass the frozen scope's stated implementation, correctness, performance, or evidence gate.

## Notes

Supersedes: HI158
Migration: capability-rebaseline-v3-2026-09
Successor key: tuning-hip-autotune-hi158

## Change Log

- 2026-09-09T10:49:14.603876+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:05:16.500617+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:00.867999+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.389969+00:00 (updated-by): Updated: section:ledger-events
