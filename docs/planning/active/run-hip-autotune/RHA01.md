---
id: RHA01
order: 0
plan: run-hip-autotune
state: pending
created-at: '2026-09-09T10:48:34.167227+00:00'
breadth: ''
skill: intermediate
created-by: capability-rebaseline-v3
work: M
priority: null
---

# profile-campaign CPU call-graph capability via perf (enhancement, deferred -- perf currently non-functional on Brutus)

## Description

Perf availability was unblocked, but perf.py integration and real-target CPU call-graph validation are explicitly not done.

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

successor-specs/run-hip-autotune-hi133.md

## Validation

Historical evidence and constraints: Preserve frozen notes/reviews/evidence on predecessor; IDs: none extracted.

Active dependencies: Frozen dependencies: none recorded.

Reference handling: Rewrite forward references (3); preserve historical references (0) on predecessor.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Close the recorded gap and pass the frozen scope's stated implementation, correctness, performance, or evidence gate.

## Notes

Supersedes: HI133
Migration: capability-rebaseline-v3-2026-09
Successor key: run-hip-autotune-hi133

## Change Log

- 2026-09-09T10:48:34.167227+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:04:37.889270+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:00.818915+00:00 (updated-by): Updated: section:ledger-events
