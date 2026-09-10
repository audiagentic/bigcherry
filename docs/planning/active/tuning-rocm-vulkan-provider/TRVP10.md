---
id: TRVP10
order: 0
plan: tuning-rocm-vulkan-provider
state: pending
created-at: '2026-09-09T11:00:52.163235+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: null
---

# CK grouped GEMM and MoE family

## Description

Evaluate grouped CK/GEMM/MoE representation only after a fail-closed schema decision separates runtime candidate-family identity from Experiment Contract taxonomy.

## Steps

1. Inventory grouped-GEMM/MoE semantic domains and existing EC16/EC19 representation.
2. Decide whether a versioned runtime family is required; do not create one for taxonomy alone.
3. If required, specify atomic schema/runtime/DB/manifest/replay changes and unavailable states.
4. Keep transformed-weight and split-K implementation in TRVP09; consume shared stack identity and provider discovery.
5. Validate conflict, persistence, replay, and provider semantics.

## Detailed Solution & Technical Design

Capability owner: tuning

Split assessment: One independent boundary; Build/Run support is a dependency.

Overlap assessment: No duplicate boundary found; related items are prerequisites or adjacent evidence.

## Code Samples & Guidance



## Files

successor-specs/tuning-rocm-vulkan-provider-ro15.md

## Validation

Historical evidence and constraints: Preserve frozen notes/reviews/evidence on predecessor.

Active dependencies: TRVP08 (RO13 successor). TRVP09 and TRVP10 are siblings; TRVP10 does not depend on TRVP09.

Reference handling: Rewrite forward references; preserve historical references on predecessor.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

A fail-closed decision records whether GROUPED_GEMM needs a runtime family; no provider implementation starts before identity/taxonomy boundary is resolved; TRVP09 remains a sibling, not a prerequisite.

## Notes

Supersedes: RO15
Migration: capability-rebaseline-v3-2026-09
Successor key: tuning-rocm-vulkan-provider-ro15

Supersedes: RO15
Inherited constraint: RV117 and RV119 — decide runtime-family versus Experiment Contract ownership first; TRVP09/TRVP10 are sibling successors after RO13.
Migration: capability-rebaseline-v3-2026-09

## Change Log

- 2026-09-09T11:00:52.163235+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:17:45.501972+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.574818+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:47.426458+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T00:52:09.973404+00:00 (updated-by): Updated: section:description, section:steps, section:acceptance_criteria, section:notes
- chg_20260910_005948_legacy-planning-folders-now-co_1240
- 2026-09-10T00:59:48.970435+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T01:02:55.006663+00:00 (updated-by): Updated: section:validation
- chg_20260910_010342_successor-plans-now-have-expli_8662
- 2026-09-10T01:03:42.557928+00:00 (updated-by): Updated: section:ledger-events
