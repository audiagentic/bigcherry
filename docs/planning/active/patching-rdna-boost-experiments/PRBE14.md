---
id: PRBE14
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:54:26.498754+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: null
---

# Evaluate MoE top-k weights folded into down projection

## Description

Evaluate MoE top-k weights folded into down projection while preserving the explicit composition boundary between patch 1207 and patch 1205.

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

successor-specs/patching-rdna-boost-experiments-rd17.md

## Validation

Historical evidence and constraints: Preserve frozen notes/reviews/evidence on predecessor; IDs: none extracted.

Active dependencies: Frozen dependencies: none recorded.

Reference handling: Rewrite forward references (4); preserve historical references (0) on predecessor.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Correctness/performance qualification covers 1207 alone. 1205 and 1207 are mutually exclusive unless an explicitly declared composed experiment passes graph identity, conflict, and composition validation through PKC02; no implicit composition is admitted.

## Notes

Supersedes: RD17
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd17

Supersedes: RD17
Inherited constraint: RV105 — retain the 1205/1207 composition conflict and mutually exclusive recipe identities; link graph/DAG composition decisions to PKC02.
Migration: capability-rebaseline-v3-2026-09

## Change Log

- 2026-09-09T10:54:26.498754+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:11:35.940393+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.192523+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.895469+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T00:52:25.957169+00:00 (updated-by): Updated: section:description, section:acceptance_criteria, section:notes
- chg_20260910_005948_legacy-planning-folders-now-co_1240
- 2026-09-10T00:59:48.994822+00:00 (updated-by): Updated: section:ledger-events
