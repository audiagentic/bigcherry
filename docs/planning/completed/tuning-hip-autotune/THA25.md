---
id: THA25
order: 0
plan: tuning-hip-autotune
state: superseded
created-at: '2026-09-09T10:50:39.551947+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: null
---

# Q8_1 activation reuse, K-tile tuning, and fused gate+up prefill for gfx1100 dense models

## Description

Investigate Q8_1 activation reuse, K-tile/barrier variants, and fused gate+up integer-MMQ prefill for gfx1100 dense models with causal, graph, and multi-GPU evidence.

## Steps

Instrument Q8_1 launches/source identity/grouping and weighted node timing; add prepacked native entry and execution-scoped planner with compatibility/lifetime; clone exact gfx1100 shapes; measure occupancy/I/thread/tile/LDS variants independently; only after stability evaluate fused gate/up prefill; require production replay evidence.

## Detailed Solution & Technical Design

Separate three hypotheses: shared activation packing, K-tile/barrier schedule, and fused integer-MMQ gate/up at N=512. Preserve native fallback and graph-safe arena lifetime; never combine candidates without attribution.

## Code Samples & Guidance



## Files

MMQ quantization wrapper/prepacked executor/planner; gfx1100 kernels; tuner/catalog/evidence.

## Validation

Q8_1 counter/identity telemetry, graph lifetime, correctness, workspace, multi-GPU and causal kernel/E2E evidence; production candidates require graph/correctness/workspace/provenance records.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Promote only independently correct, graph-safe candidates with causal performance evidence; no production replay candidate without complete evidence contract.

## Notes

Supersedes: HI26
Migration: capability-rebaseline-v3-2026-09
Successor key: tuning-hip-autotune-hi26

Consolidation disposition (2026-11-20): superseded into active THA15 after RV4250. The full family-specific preconditions and correctness/provenance gates are preserved there as conditional, unselected work. This does not claim implementation, GPU validation, candidate acceptance or promotion; native remains default. Preserve this source record and its history.

## Change Log

- 2026-09-09T10:50:39.551947+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:06:57.851550+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:00.970279+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.558248+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:33:26.685942+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria
- chg_20260910_033345_repaired-three-more-tuning-suc_6484
- 2026-09-10T03:33:45.447148+00:00 (updated-by): Updated: section:ledger-events
- 2026-11-20T03:17:56.885694+00:00 (updated-by): Updated: section:notes
- 2026-11-20T03:18:06.855621+00:00 (state-transition): State: pending → superseded
