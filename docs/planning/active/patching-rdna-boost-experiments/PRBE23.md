---
id: PRBE23
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:55:00.176506+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# AMD-MOE-001: MoE-aware MMQ tile sizing from average expert occupancy

## Description

Qualify MoE-aware MMQ work reduction using the preferred compact actual-work grid (#63 concept), with mean-occupancy selection only as a control and durable candidate identity.

## Steps

- Audit current table-driven MMQ architecture; do not port obsolete #39 mmq_x_best anchors.
- Use native rectangular grid as control, compacted expert_bounds block map as preferred treatment, and mean/expected occupancy only as explanatory control.
- Define mapping/overflow/workspace/stream-K/multi-GPU/autotune identity and preserve exact expert/tile enumeration with legacy fallback.
- Run EC13/RD94 hostile routing: uniform, Zipf/skew, concentrated, single-hot and captured natural routing at n_expert=256.
- Run record/tune/promote/replay on Qwen3.6-35B-A3B and interleaved A/B; stale candidate-tuning evidence must be remeasured after grid change.

## Detailed Solution & Technical Design

Capability owner: patching

Split assessment: One independent boundary; Build/Run support is a dependency.

Overlap assessment: No duplicate boundary found; related items are prerequisites or adjacent evidence.

## Code Samples & Guidance



## Files

patch 1237/compact-grid source and registry; mmq map/grid selector; workspace accounting; EC13/RD94 hostile routing; n_expert=256 E2E campaign; candidate identity/disposition evidence.

## Validation

Current table seam audit; exact mapping; overflow/legacy fallback; native/tune correctness; hostile routing; real dual-gfx1100 Qwen3.6-35B-A3B pp/tg; interleaved A/B; non-target controls.

## Effort & Risk



## Standards

Redesign-first; exact enumeration; fail-closed overflow; causal identity; preserve negative/noise evidence.

## Acceptance Criteria

Compact-grid candidate has durable identity and passes correctness/hostile routing; real E2E gain is reproduced with controlled A/B; no broad default or mean-based selector is promoted without its own evidence.

## Notes

Supersedes: RD30
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd30

## Change Log

- 2026-09-09T10:55:00.176506+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:12:15.436526+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events



- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.231064+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.956316+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:53:23.331205+00:00 (updated-by): Updated: section:description, section:steps, section:files, section:validation, section:standards, section:acceptance_criteria
- chg_20260910_025409_moe-mmq-successors-prbe2325-n_6205
- 2026-09-10T02:54:09.740115+00:00 (updated-by): Updated: section:ledger-events
- chg_20260911_220609_documented-the-most-thoroughly_1414
- 2026-09-11T22:06:09.771118+00:00 (updated-by): Updated: section:ledger-events
