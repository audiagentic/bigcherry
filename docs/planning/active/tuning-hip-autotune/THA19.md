---
id: THA19
order: 0
plan: tuning-hip-autotune
state: pending
created-at: '2026-09-09T10:50:14.474006+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: P3
---

# RoPE candidate-search: authorize and build tuning for RoPE dispatch once HI174 proves the pattern

## Description

RoPE candidate search is explicitly not started and is blocked on HI174 proving the non-matmul dispatch/candidate pattern.

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

successor-specs/tuning-hip-autotune-hi175.md

## Validation

Historical evidence and constraints: Preserve frozen notes/reviews/evidence on predecessor; IDs: HI67,HTR01.

Active dependencies: Frozen dependencies: HI174.

Reference handling: Rewrite forward references (9); preserve historical references (2) on predecessor.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Close the recorded gap and pass the frozen scope's stated implementation, correctness, performance, or evidence gate.

## Notes

Supersedes: HI175
Migration: capability-rebaseline-v3-2026-09
Successor key: tuning-hip-autotune-hi175

## Change Log

- 2026-09-09T10:50:14.474006+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:06:26.865897+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:00.939547+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.514013+00:00 (updated-by): Updated: section:ledger-events
