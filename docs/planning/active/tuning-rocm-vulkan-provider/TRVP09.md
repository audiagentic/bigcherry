---
id: TRVP09
order: 0
plan: tuning-rocm-vulkan-provider
state: pending
created-at: '2026-09-09T11:00:48.791946+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: null
---

# CK split-K, preshuffle, pipeline variants

## Description

Add CK split-K, preshuffle, and pipeline variants with transformed-weight identity/cost accounting and safe candidate promotion. This is a sibling of TRVP10 after the shared RO13 successor, not a dependency on grouped-GEMM.

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

successor-specs/tuning-rocm-vulkan-provider-ro14.md

## Validation

Historical evidence and constraints: Preserve frozen notes/reviews/evidence on predecessor.

Active dependencies: TRVP08 (RO13 successor). TRVP09 and TRVP10 are siblings; TRVP09 does not depend on TRVP10.

Reference handling: Rewrite forward references; preserve historical references on predecessor.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Close the recorded gap and pass the frozen scope's stated implementation, correctness, performance, or evidence gate.

## Notes

Supersedes: RO14
Migration: capability-rebaseline-v3-2026-09
Successor key: tuning-rocm-vulkan-provider-ro14

Supersedes: RO14
Inherited constraint: RV119 — retain transformed-weight ownership here and remove the erroneous dependency on TRVP10; TRVP09/TRVP10 are siblings after RO13.
Migration: capability-rebaseline-v3-2026-09

## Change Log

- 2026-09-09T11:00:48.791946+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:17:41.045081+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.569354+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:47.419247+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T00:52:01.900940+00:00 (updated-by): Updated: section:description, section:validation, section:notes
- chg_20260910_005948_legacy-planning-folders-now-co_1240
- 2026-09-10T00:59:48.958541+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T01:02:49.037323+00:00 (updated-by): Updated: section:validation
- chg_20260910_010342_successor-plans-now-have-expli_8662
- 2026-09-10T01:03:42.545695+00:00 (updated-by): Updated: section:ledger-events
