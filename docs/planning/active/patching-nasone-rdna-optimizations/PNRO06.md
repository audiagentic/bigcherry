---
id: PNRO06
order: 0
plan: patching-nasone-rdna-optimizations
state: pending
created-at: '2026-09-09T10:52:38.138969+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: P0
---

# Hybrid ROCm TOP_K kernels

## Description

Port and qualify nasone hybrid HIP TOP_K selection as an independent foundation before wave32 tuning. Preserve CUDA/CUB behavior and native fallback.

## Steps

- Freeze source 7f3e1e4d... and audit dispatch against b10705 top-k.cu.
- Add HIP-only TOP_K selection kernels with ordered-float NaN/Inf/tie policy, TOP-1 reduction, n-ary/radix paths and bitonic fallback for unsupported shapes.
- Keep selection opt-in/traceable; do not invent a matmul candidate ID or replace HIP TOP_K globally before exact-index correctness.
- Build CPU/reference fixtures for k/nrows/ncols, ties, negative values, infinities and duplicates; verify caller ordering semantics.
- Capture real MoE/QSA signatures and compare hybrid against bitonic at k=1 and routing k=2/4/8/10; keep PNRO07 disabled for causal attribution.

## Detailed Solution & Technical Design

The optimization changes algorithmic complexity and temporary storage, not just geometry. Exact index/tie semantics are load-bearing; any stable-order divergence requires an explicit policy. Keep HIP-only containment and native fallback.

## Code Samples & Guidance



## Files

patches/1256_nro07_topk_hybrid/{patch.toml,patch.py,SUMMARY.md,README.md,TESTING.md}; top-k.cu; shared NRO static tests; CPU/reference TOP_K fixture runner; real signature campaign artifacts.

## Validation

Non-HIP preprocessor preservation; idempotent apply; exact index/order fixtures k=1..ncols and multi-row; unsupported fallback; gfx1100 real MoE/QSA signatures; kernel/scratch/end-to-end call-weighted performance.

## Effort & Risk



## Standards

Exact routing correctness before performance; HIP-only containment; native fallback; no conflation with downstream routing fusion.

## Acceptance Criteria

Reference and routing correctness pass; non-HIP builds unchanged; unsupported shapes fall back; at least one real gfx1100 high-cost signature shows repeatable win without model regression; PNRO07 remains separate.

## Notes

Supersedes: NRO07
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-nasone-rdna-optimizations-nro07

## Change Log

- 2026-09-09T10:52:38.138969+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:08:57.337897+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.075875+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.719952+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:42:40.812059+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria
- chg_20260910_024304_three-nasone-successors-now-pr_2691
- 2026-09-10T02:43:04.858380+00:00 (updated-by): Updated: section:ledger-events
