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

Qualify patch 1205 paired MMVQ matmuls over shared activation with RD25 correctness state and composition-conflict controls.

## Steps

- Resolve RD25 batch-vs-seq consistency as a hard prerequisite and verify patch 1205 source identity/post-image.
- Detect only exact K/V paired MMVQ graph patterns with shared activation lifetime and same output shape.
- Preserve GLU fusion precedence, views/no-ops, false-positive fallback and unfused reference.
- Treat patch 1205 and 1207 as composition-conflicting unless a dedicated recipe declares order and validates both; include PRBE05 only as explicit identity dependency.
- Run isolated rd12-only first on native plus RD25, then graph capture, correctness and causal performance with real signatures.

## Detailed Solution & Technical Design

Use a graph rewrite only after proof of shared activation lifetime. Patch identity must include RD25 and any declared composition; no hidden cache/fusion changes. 1205/1207 overlap common slots by design and cannot be combined implicitly.

## Code Samples & Guidance



## Files

patches/1205_rd12_paired_mmvq_dual_output; RD25 correctness state; graph planner/MMVQ seam; exact-pattern/fallback fixtures; rd12-only and declared-composition evidence.

## Validation

K/V pairs; same shape; GLU precedence; views/no-ops; false positives; unfused numerical reference; graph capture; isolated and declared composition causal arms.

## Effort & Risk



## Standards

Exact graph pattern; dependency-aware comparison; fallback; isolated-test-first.

## Acceptance Criteria

Exact patterns are correct with RD25; unsafe/near-miss graphs fall back; isolated performance and any composition result are separately attributable; 1207 is not silently combined.

## Notes

Supersedes: RD12
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd12

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
