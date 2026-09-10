---
id: PRBE21
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:54:53.078402+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: null
---

# Fold SSM conv_input concat into qkv mmvq; rpb=2 for small-K MoE

## Description

Qualify SSM conv_input folding into qkv MMVQ and rpb=2 small-K MoE as an isolated candidate with dependency audit.

## Steps

- Audit exact Q8/SSM symbols against existing 0600/0650/common.cuh and determine PRBE05/PRBE18 relationships explicitly.
- Port source 0510d7cfa as an rdna-boosts package with provenance/registry; do not assume cache or SSM-chain dependency.
- Run isolated SSM/Mamba fused-vs-unfused correctness and timing on gfx1100.
- Check exact shapes, memory/bounds, false-pattern fallback and small-K MoE rpb behavior.
- Only evaluate composition after isolated candidate is proven and dependency identity is declared.

## Detailed Solution & Technical Design

Capability owner: patching

Split assessment: One independent boundary; Build/Run support is a dependency.

Overlap assessment: No duplicate boundary found; related items are prerequisites or adjacent evidence.

## Code Samples & Guidance



## Files

common.cuh; ggml-cuda.cu; mmvq.cu; patch 12xx rd27; external source entry; SSM/Mamba fixtures; fused/unfused and fallback campaign artifacts.

## Validation

Source/anchor audit; output equality; exact SSM/Mamba shapes; gfx1100; memory/bounds; small-K rpb; false-pattern fallback; causal timing.

## Effort & Risk



## Standards

Dependency audit; exact pattern; memory safety; isolated qualification.

## Acceptance Criteria

Fused path is correct for exact SSM/small-K patterns, unsupported patterns fall back, and any benefit is shown in isolated causal evidence before composition.

## Notes

Supersedes: RD27
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd27

## Change Log

- 2026-09-09T10:54:53.078402+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:12:06.531616+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.222985+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.942391+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:51:58.669575+00:00 (updated-by): Updated: section:description, section:steps, section:files, section:validation, section:standards, section:acceptance_criteria
- chg_20260910_025218_rdna-successors-prbe2022-now_5714
- 2026-09-10T02:52:18.419086+00:00 (updated-by): Updated: section:ledger-events
