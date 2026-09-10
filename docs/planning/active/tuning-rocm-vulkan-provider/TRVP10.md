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

Resolve the grouped-GEMM/MoE schema boundary and, only if required, add a versioned GROUPED_GEMM runtime family with provider, persistence, replay, and correctness semantics.

## Steps

1. Inventory grouped-GEMM/MoE semantic domains and current EC16/EC19 Experiment Contract representations. 2. Make a fail-closed decision whether runtime candidate-family identity cannot be represented by the existing families; do not create a family for taxonomy alone. 3. If required, update family enums, schema, registry, database CHECK constraints, coverage/reporting, manifests, and runtime/replay contracts atomically, including unavailable states and migration compatibility. 4. Keep split-K/preshuffle/transformed-weight ownership in TRVP09 and consume shared provider/stack identity. 5. Enumerate CK grouped candidates and validate expert routing, permutations, empty groups, workspace, provider identity, fallback, graph/inference behavior, independent oracle, and promotion/replay.

## Detailed Solution & Technical Design

Grouped CK/MoE must not be smuggled through dense BLAS eligibility, which rejects MUL_MAT_ID. First record the EC16/EC19 versus runtime-family decision. If a true runtime family is needed, add GROUPED_GEMM across enums, schema, registry, DB constraints, coverage/reporting, manifests, serialization, and replay as one compatibility-aware change; represent unsupported/unavailable states explicitly. Provider discovery stays here, while transformed-weight and split-K work stays in TRVP09. Candidate identity uses shared exact CK/provider/build identity.

## Code Samples & Guidance



## Files

hip-autotune-types.h; tuning schema/catalog/registry; DB constraints and migrations; coverage/report tooling; grouped provider/runtime route; manifest/replay contracts; correctness and schema compatibility tests.

## Validation

Inventory and decision artifact names every grouped domain and current EC16/EC19 representation. If a new family is introduced, verify all schema/DB/registry/report/replay contracts agree and old data migrates safely. Run grouped correctness with empty groups/experts, all tokens to one expert, skew, repeated experts, non-contiguous inputs, padded shapes, metadata ordering/maxima, deterministic seeds, independent oracle, tolerances, workspace, fallback, and graph/inference checks. Confirm dense candidates cannot accept grouped work and grouped identity cannot replay across providers/builds.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

A fail-closed runtime-family decision is recorded before implementation. No grouped CK/MoE work enters dense BLAS. If GROUPED_GEMM is required, all schema, DB, registry, manifest, coverage, and replay changes land atomically with unavailable states and compatibility tests. Grouped routing/correctness/identity/fallback/promotion gates pass; TRVP09 remains a sibling.

## Notes

Supersedes: RO15
Migration: capability-rebaseline-v3-2026-09
Successor key: tuning-rocm-vulkan-provider-ro15

Supersedes: RO15
Inherited constraint: RV117 and RV119 — decide runtime-family versus Experiment Contract ownership first; TRVP09/TRVP10 are sibling successors after RO13.
Migration: capability-rebaseline-v3-2026-09

Supersedes RO15. Preserve RV117/RV119: resolve runtime-family versus EC ownership first, and do not make TRVP09 a prerequisite. Preserve patch 1225 and ledger governance.

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
- 2026-09-10T03:45:11.796331+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria, section:notes
- chg_20260910_034539_repaired-trvp10-12-with-the-co_1657
- 2026-09-10T03:45:39.433583+00:00 (updated-by): Updated: section:ledger-events
