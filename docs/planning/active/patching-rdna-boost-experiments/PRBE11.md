---
id: PRBE11
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:54:15.041193+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: null
---

# Evaluate paired MMVQ matmuls over shared activation

## Description

Qualify patch 1205 paired MMVQ matmuls over shared activation with explicit composition-conflict controls. PRBE11 is the authoritative active successor to closed RD12 and is independent of PRBE19's RD25 bake-in rule.

## Steps

1. Verify patch 1205 source identity/post-image and detect only exact K/V paired MMVQ graph patterns with shared activation lifetime, same output shape, compatible op/type gates, and safe view/data intervals.
2. Preserve GLU fusion precedence, views/no-ops, false-positive fallback and unfused reference.
3. Preserve the paired implementation details: shared source allocation/overlap detection, disjoint outputs, shared quantized-X launch, and runtime grid adjustment.
4. Treat patch 1205 and 1207 as composition-conflicting unless a dedicated recipe declares order and validates both. Include PRBE05 only when an actual source audit proves cache identity dependence.
5. Run isolated PRBE11 first on native state, then declared compositions, graph capture, correctness and causal performance.

## Detailed Solution & Technical Design

Use a graph rewrite only after proof of shared activation lifetime and exact source/view interval relationships. Patch identity must include the declared composition, but must not invent an RD25 dependency: historical RD25 does not touch the paired-MMVQ implementation. 1205/1207 overlap common slots by design and cannot be combined implicitly. Preserve the native and unfused fallback.

## Code Samples & Guidance



## Files

patches/1205_rd12_paired_mmvq_dual_output; graph planner/MMVQ seam; exact-pattern/fallback fixtures; paired reshape/view interval tests; PRBE05 only if source-audit evidence requires it; 1207 composition evidence.

## Validation

K/V pairs; same shape; same-source overlap/disjoint-output gates; GLU precedence; views/no-ops; false positives; unfused numerical reference; graph capture; isolated and declared-composition causal arms.

## Effort & Risk



## Standards

Exact graph pattern; dependency-aware comparison; fallback; isolated-test-first.

## Acceptance Criteria

Exact patterns are correct with RD25; unsafe/near-miss graphs fall back; isolated performance and any composition result are separately attributable; 1207 is not silently combined.

## Notes

Supersedes: RD12
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd12
Supersedes: RD12 (closed historical predecessor). Preserve source identities 44b51c663... / upstream ba9e339ea... as provenance. PRBE19 is explicitly non-applicable; do not apply raw RD25 or describe it as a prerequisite. PRBE11 may consume PRBE05 only after a real source audit establishes that dependency.

## Change Log

- 2026-09-09T10:54:15.041193+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:11:22.576195+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.177979+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.874477+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:47:35.743786+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria
- chg_20260910_024759_three-rdna-fusion-successors-n_3469
- 2026-09-10T02:47:59.451164+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-11T16:41:04.517699+00:00 (updated-by): Updated: section:notes
- chg_20260911_164110_investigated-why-an-experiment_2332
- 2026-09-11T16:41:10.876945+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-12T09:52:15.950084+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:notes
- chg_20260912_095506_cleaned-the-active-rdna-boost_4906
- 2026-09-12T09:55:06.207567+00:00 (updated-by): Updated: section:ledger-events
