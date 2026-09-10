---
id: PRBE12
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:54:19.359011+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# Evaluate MUL_MAT plus RESHAPE plus ADD fusion

## Description

Qualify patch 1206 MUL_MAT+RESHAPE+ADD view fusion after offline safety repair, including CUDA equivalence, capture and causal performance.

## Steps

- Use only the exact RESHAPE-mediated view/add pattern and ggml_can_fuse_subgraph safety checks.
- Require view specifically at ADD.src[0], reject null addends, wrong wiring, extra consumers, non-VIEW nodes and direct-ADD near misses.
- Compare fused/unfused outputs and graph capture/replay on CUDA/HIP where supported.
- Keep PRBE05/Q8 cache and other enhancements out of the standalone arm unless explicitly declared in identity.
- Run balanced timing only after correctness and capture gates pass.

## Detailed Solution & Technical Design

The new view-mediated matcher is non-commutative by safety contract; preserve legacy direct-ADD matcher behavior unchanged. No false-positive graph rewrite is acceptable.

## Code Samples & Guidance



## Files

patches/1206_rd13_mul_mat_add_view_fusion; graph matcher/fusion source; targeted safety tests; fused/unfused output fixtures; graph capture and timing artifacts.

## Validation

Direct ADD exclusion; view at src[0]; reversed wiring; null/extra-consumer/non-VIEW rejection; output parity; graph capture/replay; balanced causal timing.

## Effort & Risk



## Standards

Exact pattern; no false positives; causal isolation; preserve legacy direct-ADD semantics.

## Acceptance Criteria

All exact-pattern and negative fixtures pass; fused output matches unfused/reference; graph capture/replay is stable; only a statistically supported benefit without regressions is promotable.

## Notes

Supersedes: RD13
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd13

## Change Log

- 2026-09-09T10:54:19.359011+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:11:26.671901+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.183883+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.881480+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:47:41.484974+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria
- chg_20260910_024759_three-rdna-fusion-successors-n_3469
- 2026-09-10T02:47:59.468336+00:00 (updated-by): Updated: section:ledger-events
