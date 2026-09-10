---
id: PRBE03
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:53:39.131985+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: null
---

# Evaluate RDNA4 expanded WMMA flash-attention configurations

## Description

Qualify the materialized gfx1201-only expanded WMMA flash-attention configuration after PRBE02 correctness barriers pass. The treatment must not select or regress gfx1100.

## Steps

- Resolve baseline plus PRBE02, then apply only the expanded RDNA4 configuration hunk from patch 1203.
- Run per-shape and softcap cases on gfx1201 with balanced repeated treatment/control measurements.
- Inspect resource use and occupancy and compare outputs to the correctness-qualified baseline.
- Run explicit gfx1100 non-selection/non-regression controls and retain architecture rejection evidence.
- Record resolved identity, environment, shape matrix, repetitions and promotion decision.

## Detailed Solution & Technical Design

Use a campaign arm whose sole intended difference is the expanded RDNA4 configuration. Keep native BF16 and the PRBE02 correctness prerequisite explicit in the resolved identity; no cross-architecture default is permitted.

## Code Samples & Guidance



## Files

patches/1203_rd050607_rdna4_wmma_fa_q6k_mmq; gfx1201 WMMA configuration; per-shape/softcap campaign artifacts; occupancy reports; gfx1100 control evidence.

## Validation

gfx1201 isolated benchmark; per-shape head dimensions and softcap cases; resource/occupancy inspection; output correctness; gfx1100 non-selection and non-regression; balanced repeats with confidence intervals.

## Effort & Risk



## Standards

Causal comparison; architecture gating; correctness prerequisite; no blanket default from a single topology.

## Acceptance Criteria

Only the correctness-qualified gfx1201 arm is evaluated; every selected shape passes output checks and the registered performance/evidence gate; gfx1100 cannot select it and shows no regression; otherwise reject/quarantine the configuration.

## Notes

Supersedes: RD06
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd06

## Change Log

- 2026-09-09T10:53:39.131985+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:10:18.220021+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.138725+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.811061+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:30:18.912456+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria
- chg_20260910_023037_three-rdna-boost-successors-no_6965
- 2026-09-10T02:30:37.084094+00:00 (updated-by): Updated: section:ledger-events
