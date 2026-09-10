---
id: PHC02
order: 0
plan: patching-hip-collectives
state: superseded
created-at: '2026-09-09T10:51:17.291510+00:00'
breadth: ''
skill: ''
created-by: capability-rebaseline-v3
priority: null
---

# Validate META SPLIT_REDUCE correctness for D=3/D=4 heterogeneous topologies

## Description

RCCL arm is superseded, but META D=3/D=4 qualification remains unfinished.

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

successor-specs/patching-hip-collectives-hi84.md

## Validation

Historical evidence and constraints: Preserve frozen notes/reviews/evidence on predecessor; IDs: HI138,HI18,HI58,HI85.

Active dependencies: Frozen dependencies: none recorded.

Reference handling: Rewrite forward references (2); preserve historical references (4) on predecessor.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Close the recorded gap and pass the frozen scope's stated implementation, correctness, performance, or evidence gate.

## Notes

Supersedes: HI84
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-hip-collectives-hi84

Superseded by: RU01 (run-hip-collectives)
Migration: capability-rebaseline-v3-2026-09
Reason: RV115 confirms this is Run/evidence work, not patch lifecycle; preserve PHC02 as historical migration record.

## Change Log

- 2026-09-09T10:51:17.291510+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:07:38.122818+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.007309+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.611075+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T00:53:34.140328+00:00 (updated-by): Updated: section:notes
- 2026-09-10T00:53:41.813084+00:00 (state-transition): State: pending → superseded
- chg_20260910_005948_legacy-planning-folders-now-co_1240
- 2026-09-10T00:59:49.082193+00:00 (updated-by): Updated: section:ledger-events
