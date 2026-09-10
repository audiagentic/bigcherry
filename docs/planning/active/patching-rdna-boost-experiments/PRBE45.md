---
id: PRBE45
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:56:36.141433+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# AMD-GDN-004: Launch-bounds/VGPR occupancy tuning for chunked GDN

## Description

Tune launch bounds and VGPR occupancy for the PRBE42 chunked GDN kernel per architecture, preventing register spills while preserving the baseline on non-target generations.

## Steps

1. Characterize PRBE42 kernel signatures, VGPR/LDS use, occupancy, and spill behavior by gfx target. 2. Evaluate explicit launch bounds/occupancy settings, starting with gfx1151 source targets and separately tuning gfx1100/gfx1201. 3. Keep all non-target architectures and unsupported signatures on the baseline settings. 4. Add unchanged-output/state tests and inspect generated code for spills and occupancy. 5. Benchmark kernel microseconds and E2E prefill against PRBE42 baseline; promote only an architecture-specific winner.

## Detailed Solution & Technical Design

Apply explicit register/occupancy constraints to the chunked GDN kernel so enough blocks remain resident and register-resident recurrent state does not spill. Treat gfx1151 as a source-informed target but require independent measurements for other RDNA generations. Selection must be architecture-specific and preserve baseline launch bounds when no win is demonstrated.

## Code Samples & Guidance

Trigger: supported PRBE42 GDN signatures on the tuned architecture. Controls: all non-target architectures, unsupported signatures, and baseline launch configuration. Boundary: occupancy target, launch bounds, VGPR/LDS and spill thresholds.

## Files

PRBE42 GDN launch-bound attributes/configuration and architecture selector; output/state parity tests; compiler occupancy/spill reports; per-architecture kernel and PP replay evidence for AMD-GDN-004.

## Validation

Correctness: unchanged output and recurrent state against baseline. Performance: report kernel microseconds, occupancy, VGPR/LDS, spills, and PP E2E with variance. Acceptance: enable only an architecture-specific winner with repeatable benefit and no spill/correctness/end-to-end regression; retain baseline otherwise.

## Effort & Risk

M; launch bounds can trade occupancy against spills or underfill. Require generated-code inspection and per-architecture evidence.

## Standards

Preserve PRBE42 eligibility/fallback, target-specific selection, deterministic output, and campaign evidence provenance.

## Acceptance Criteria

Acceptance requires unchanged output/state, no unacceptable VGPR/LDS spills, per-architecture occupancy evidence, and a repeatable kernel/E2E benefit without regression; otherwise retain baseline launch bounds.

## Notes

Supersedes: RD53
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd53

Supersedes RD53. Depends on PRBE42 and is last in the AMD-GDN prerequisite chain; do not tune outside the chunked kernel.

## Change Log

- 2026-09-09T10:56:36.141433+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:13:50.339230+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.332298+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.103893+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:07:15.775864+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:effort_risk, section:standards, section:notes
- 2026-09-10T03:07:35.048095+00:00 (updated-by): Updated: section:acceptance_criteria
- chg_20260910_030747_carried-forward-the-remaining_4294
- 2026-09-10T03:07:47.489283+00:00 (updated-by): Updated: section:ledger-events
