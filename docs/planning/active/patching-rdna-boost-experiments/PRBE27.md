---
id: PRBE27
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:55:19.101435+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# AMD-MMQ-002: Dedicated RDNA3.5 MMQ device table

## Description

The old gfx1151 MMQ diff no longer applies; the item remains re-scoped to checking/redesigning the current table-driven configuration with hardware evidence.

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

successor-specs/patching-rdna-boost-experiments-rd34.md

## Validation

Historical evidence and constraints: Preserve frozen notes/reviews/evidence on predecessor; IDs: none extracted.

Active dependencies: Frozen dependencies: none recorded.

Reference handling: Rewrite forward references (2); preserve historical references (0) on predecessor.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Close the recorded gap and pass the frozen scope's stated implementation, correctness, performance, or evidence gate.

## Notes

Supersedes: RD34
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd34

## Change Log

- 2026-09-09T10:55:19.101435+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:12:31.554843+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.249087+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.981923+00:00 (updated-by): Updated: section:ledger-events
