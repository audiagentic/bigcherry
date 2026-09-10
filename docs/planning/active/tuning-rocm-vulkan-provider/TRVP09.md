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

Add CK split-K, preshuffle, scheduler/pipeline variants with transformed-weight identity, preparation-cost accounting, and safe correctness-gated promotion.

## Steps

1. Build on TRVP08 dense CK provider identity; keep TRVP09 and TRVP10 as siblings, not a serialized dependency. 2. Add split-K, preshuffled GEMM, scheduler, and pipeline candidate dimensions. 3. Model transformed-weight identity as original hash plus transform/version, provider/architecture, and transformed hash. 4. Account for one-time transform/conversion cost, VRAM/memory impact, and steady-state time in promotion decisions; steady-state speed cannot hide preparation cost. 5. Define deterministic seeded correctness matrices for split-K 1/max/invalid, tile/K remainders, workspace reuse, accumulation tolerance, preshuffle format/version, and pipeline identity. 6. Use an independent oracle and exact provider/build identity before promotion; invalidate transformed artifacts on provider/config changes.

## Detailed Solution & Technical Design

The CK variant catalog extends the dense provider with explicit split-K, preshuffle, scheduler, and pipeline dimensions. Transformed artifacts are immutable, versioned, and namespaced by original weight hash, transform/version, provider, architecture, and transformed hash. Candidate records include preparation time, memory/VRAM, steady-state timing, workspace, seed, tolerances, and stack/build fingerprint. Any provider/config/format change invalidates stale artifacts. Promotion requires correctness for split-K edge cases and independent comparison, with deterministic fallback to dense/native when unsupported.

## Code Samples & Guidance



## Files

CK provider/catalog/runtime variant implementation; transformed-weight artifact schema/cache/invalidation; split-K/preshuffle/pipeline tests; independent correctness oracle; qualification reports.

## Validation

Run split-K 1/max/invalid and tile/K remainder matrices, workspace reuse, accumulation-tolerance checks, preshuffle format/version checks, and pipeline identity checks with deterministic seeds and documented tolerances. Measure one-time preparation/conversion cost, VRAM impact, and steady-state time together. Verify stale transformed artifacts invalidate on provider/config changes and unsupported variants fall back safely. Require exact provider/build identity and independent oracle evidence before promotion.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Variant candidates are provider/build/architecture/format identifiable and reproducible. Correctness passes for split-K boundaries, remainders, workspace reuse, accumulation tolerance, preshuffle, and pipeline identity with documented tolerances. Promotion includes preparation cost and memory impact; stale transformed artifacts are invalidated; unsupported variants fail safely to dense/native. TRVP09 remains sibling to TRVP10.

## Notes

Supersedes: RO14
Migration: capability-rebaseline-v3-2026-09
Successor key: tuning-rocm-vulkan-provider-ro14

Supersedes: RO14
Inherited constraint: RV119 — retain transformed-weight ownership here and remove the erroneous dependency on TRVP10; TRVP09/TRVP10 are siblings after RO13.
Migration: capability-rebaseline-v3-2026-09

Supersedes RO14. Preserve RV119: transformed-weight ownership stays here and TRVP09 must not depend on TRVP10. Preserve patch 1225 and ledger/planning governance.

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
- 2026-09-10T03:44:03.460810+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria, section:notes
- chg_20260910_034415_repaired-trvp07-09-with-the-co_7761
- 2026-09-10T03:44:15.631842+00:00 (updated-by): Updated: section:ledger-events
