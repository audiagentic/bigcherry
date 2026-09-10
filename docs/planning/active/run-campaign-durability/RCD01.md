---
id: RCD01
order: 0
plan: run-campaign-durability
state: pending
created-at: '2026-09-09T10:47:10.667188+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: null
---

# Durable campaign restart/resume (deferred from reusable-build-campaign RE11)

## Description

Deferred RE11 durability protocol remains explicitly retained; cross-process run resume is unfinished.

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

successor-specs/run-campaign-durability-cd01.md

## Validation

Historical evidence and constraints: Preserve frozen notes/reviews/evidence on predecessor; IDs: HI82,RE11,RE15,RE23,RE25,docs/planning/active/reusable-build-campaign/RE11.md.

Active dependencies: Frozen dependencies: none recorded.

Reference handling: Rewrite forward references (1); preserve historical references (6) on predecessor.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Close the recorded gap and pass the frozen scope's stated implementation, correctness, performance, or evidence gate.

## Notes

Supersedes: CD01
Migration: capability-rebaseline-v3-2026-09
Successor key: run-campaign-durability-cd01

External dev-gpt holistic review (2026-09-10, req_9f60aaa2ae5f4b88): CONFIRMED PARK / do not implement now. Predecessor design is sufficient as a dormant reference; add one explicit activation trigger not previously stated: reactivate only after repeated REAL campaign failures that actually require cross-process/host recovery (not preemptively). Before any eventual implementation, resolve cross-host fencing/ownership (how a new host proves the former executor cannot publish while cross-host lock-breaking remains forbidden). Preserve the RE11 design's operation-spec vs execution-hash split, immutable result records, byte verification, and interrupted-never-equals-success rule. Execution order: explicitly NOT ranked in the 1-12 sequence — stays dormant.

## Change Log

- 2026-09-09T10:47:10.667188+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:03:16.931644+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:00.731253+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T00:07:59.576114+00:00 (updated-by): Updated: section:notes
- chg_20260910_000828_reviewed-and-re-planned-all-pe_3612
- 2026-09-10T00:08:28.898595+00:00 (updated-by): Updated: section:ledger-events
