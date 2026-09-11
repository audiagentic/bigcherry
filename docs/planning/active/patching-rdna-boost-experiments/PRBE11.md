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



DEEPER BLOCKER DISCOVERED (2026-09-11/12): attempted to unblock this item by porting RD25 (its declared hard prerequisite) for real. Fetched RD25's real commit (8cdf1ab081384aa1786bfff3a45e0ec341f9fd52, stew675-rdna-boosts, branch rdna-boosts) from the tracked fork -- a real, bounded 104-line diff to ggml-cuda/mmvq.cu. But RD25's diff MODIFIES three kernel functions (ssm_gate_beta_fused_q8_0, ssm_conv_l2_gatebeta_fused, shexp_down_gated_q8_0) that do NOT EXIST anywhere in this project's current pinned vendor tree, nor in any existing or planned BigCherry patch (grep-verified against both, real search, not assumption).

Root cause found via git log -S (pickaxe search) against the fork's full history: those three kernels were introduced by a DIFFERENT, completely untracked fork commit -- 5efcd85fb4cd8845c6c7dd47c50e2666264aa4eb, "rdna-boosts: block 08: fused-core prefill kernels and GPU bit-identical" -- which is NOT in config/external-sources.toml's tracked list at all. This is a real, substantial, previously-unknown prerequisite: 1585 insertions across 10 files (common.cuh, fattn-tile.cuh, fattn.cu, ggml-cuda.cu +527, mmvq.cu +670, mmvq.cuh, norm.cu/.cuh, unary.cu/.cuh) -- an order of magnitude larger than RD25 itself, and has never been reviewed, tracked, or assessed by this project.

DECISION: did not attempt to port "block 08" or RD25 in this session. Forcing either through with guessed/unreviewed anchors against a 1585-line untracked commit this project has never even audited would violate the real rigor this project's own patch-authoring standards require (verified anchors, no invented provenance, real review before code). This is genuinely new, substantial work -- first track+audit "block 08" via `python -m bigcherry sources check`-style real review (a new [[sources.tracked]] entry, real commit content review, real scope assessment of what it does and whether it's even wanted), THEN port it as its own dedicated patch (likely several patch-sized pieces given its size), THEN port RD25 on top, THEN this item's own RD12 qualification work.

PRBE11 (and therefore RD12) remains genuinely blocked -- now documented with the REAL, complete dependency chain rather than the previously-understated "needs RD25" note. Filed as new prerequisite scope here rather than a new plan item, since it is squarely in this item's own critical path.

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
