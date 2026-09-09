---
id: PRBE03
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:53:39.131985+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: null
---

# Evaluate RDNA4 expanded WMMA flash-attention configurations

## Description

Patch 1203 materialized the RDNA4 WMMA configuration, but the frozen item explicitly awaits isolated gfx1201 benchmarking.

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

successor-specs/patching-rdna-boost-experiments-rd06.md

## Validation

Historical evidence and constraints: Preserve frozen notes/reviews/evidence on predecessor; IDs: RD02,RD03.

Active dependencies: Frozen dependencies: RD02,RD03,RD05.

Reference handling: Rewrite forward references (2); preserve historical references (2) on predecessor.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Close the recorded gap and pass the frozen scope's stated implementation, correctness, performance, or evidence gate.

## Notes

Supersedes: RD06
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd06

## Change Log

- 2026-09-09T10:53:39.131985+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:10:18.220021+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.138725+00:00 (updated-by): Updated: section:ledger-events
