---
id: RGC01
order: 0
plan: run-gpu-collectives
state: pending
created-at: '2026-09-09T10:47:57.287397+00:00'
breadth: ''
skill: intermediate
created-by: capability-rebaseline-v3
work: S
priority: null
---

# Pin MTP head tensors to a single GPU via --override-tensor to avoid cross-GPU AllReduce on the draft path

## Description

The naive tensor override produced a negative result, but the source explicitly leaves split-mode discrimination and possible graph-builder work open.

## Steps

1. Implement the still-valid future scope.
2. Run the stated acceptance and evidence gates.
3. Preserve predecessor provenance and record successor evidence under this ID.

## Detailed Solution & Technical Design

Capability owner: run

Split assessment: One independent boundary; Build/Run support is a dependency.

Overlap assessment: No duplicate boundary found; related items are prerequisites or adjacent evidence.

## Code Samples & Guidance



## Files

successor-specs/run-gpu-collectives-gp12.md

## Validation

Historical evidence and constraints: Preserve frozen notes/reviews/evidence on predecessor; IDs: GP03.

Active dependencies: Frozen dependencies: none recorded.

Reference handling: Rewrite forward references (2); preserve historical references (1) on predecessor.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Close the recorded gap and pass the frozen scope's stated implementation, correctness, performance, or evidence gate.

## Notes

Supersedes: GP12
Migration: capability-rebaseline-v3-2026-09
Successor key: run-gpu-collectives-gp12

## Change Log

- 2026-09-09T10:47:57.287397+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:03:56.793204+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:00.775870+00:00 (updated-by): Updated: section:ledger-events
