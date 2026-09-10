---
id: PNRO12
order: 0
plan: patching-nasone-rdna-optimizations
state: pending
created-at: '2026-09-09T10:53:04.447530+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: P1
---

# Qwen4exp gather-based sparse QSA decode

## Description

Resolve upstream ancestry/backport and qualify Qwen4exp gather-based sparse QSA decode; do not duplicate an equivalent upstream implementation.

## Steps

- Resolve PR #28213/equivalent head and ancestry against current/candidate pin before authoring a patch.
- If absent, port with QSA/decode predicate, n_kv>=4*width threshold, runtime escape hatch, and selected K/V/bias indices preserving rotation/cache semantics.
- Validate padded top-k width, -inf masking and single-token-per-stream restriction; nonqualifying contexts retain masked full-cache path.
- Sweep below/at/above threshold and compression ratios; compare selected-index and attention outputs to masked reference.
- Measure KV bytes, gather/dequant/cast cost, FA cost and total decode; retire local patch when equivalent becomes baseline.

## Detailed Solution & Technical Design

Gathering bounds attention work near indexer_top_k+ratio but adds GET_ROWS/dequant overhead, so a crossover selector is required. Bias and position/stream semantics are load-bearing; model-specific Qwen4exp scope must not generalize to ordinary attention.

## Code Samples & Guidance



## Files

Qwen4exp graph/model declarations; graph-input mask setup; upstream ancestry record; gather implementation/patch if absent; exact attention/reference and long-context campaign evidence.

## Validation

Ancestry; selected-index equality; exact/near-exact attention; threshold boundary; multi-stream; quantized KV; long-context memory/latency; nonqualifying original path.

## Effort & Risk



## Standards

Upstream-before-backport; attention correctness first; explicit crossover selector; no ordinary-attention generalization.

## Acceptance Criteria

Correctness matches masked QSA over threshold matrix; nonqualifying contexts retain original path; long-context decode shows positive effect after gather overhead; local patch is retired once upstream equivalent is baseline.

## Notes

Supersedes: NRO13
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-nasone-rdna-optimizations-nro13

## Change Log

- 2026-09-09T10:53:04.447530+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:09:32.138424+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.103800+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.758828+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:45:42.383626+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria
- chg_20260910_024630_the-remaining-nasone-successor_5195
- 2026-09-10T02:46:30.038448+00:00 (updated-by): Updated: section:ledger-events
