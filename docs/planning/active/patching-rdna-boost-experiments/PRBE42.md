---
id: PRBE42
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:56:23.562520+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: null
---

# AMD-GDN-001: Chunked fused GatedDeltaNet recurrence

## Description

Frozen RD50 records compile safety and fallback evidence but explicitly leaves the gfx1151/RDNA3.5 performance claim blocked and unvalidated.

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

successor-specs/patching-rdna-boost-experiments-rd50.md

## Validation

Historical evidence and constraints: Preserve frozen notes/reviews/evidence on predecessor; IDs: HI82,RD90.

Active dependencies: Frozen dependencies: RD51,RD52,RD53,RD90.

Reference handling: Rewrite forward references (4); preserve historical references (2) on predecessor.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Close the recorded gap and pass the frozen scope's stated implementation, correctness, performance, or evidence gate.

## Notes

Supersedes: RD50
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd50

## Change Log

- 2026-09-09T10:56:23.562520+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:13:38.267576+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.319945+00:00 (updated-by): Updated: section:ledger-events
