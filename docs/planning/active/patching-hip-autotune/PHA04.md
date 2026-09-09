---
id: PHA04
order: 0
plan: patching-hip-autotune
state: pending
created-at: '2026-09-09T10:48:57.946592+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# RCCL admission gate: implement level-2 per-call candidate binding

## Description

Implementation, tests, and hardware validation have not started.

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

successor-specs/patching-hip-autotune-hi147.md

## Validation

Historical evidence and constraints: Preserve frozen notes/reviews/evidence on predecessor; IDs: HI85.

Active dependencies: Frozen dependencies: HI142,HI145,HI146.

Reference handling: Rewrite forward references (4); preserve historical references (1) on predecessor.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Close the recorded gap and pass the frozen scope's stated implementation, correctness, performance, or evidence gate.

## Notes

Supersedes: HI147
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-hip-autotune-hi147

## Change Log

- 2026-09-09T10:48:57.946592+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:05:02.519090+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:00.848546+00:00 (updated-by): Updated: section:ledger-events
