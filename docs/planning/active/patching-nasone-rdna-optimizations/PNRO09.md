---
id: PNRO09
order: 0
plan: patching-nasone-rdna-optimizations
state: pending
created-at: '2026-09-09T10:52:50.276971+00:00'
breadth: ''
skill: intermediate
created-by: capability-rebaseline-v3
work: S
priority: P1
---

# Increase Meta compute-container view headroom for recurrent/MTP graphs

## Description

Evaluate Meta compute-container view headroom for recurrent/MTP graphs. Port only after proving the existing 16-view bound is insufficient or a supported graph-derived bound exceeds it.

## Steps

- Confirm b10705 Meta allocator behavior and derive actual maximum static-tensor view count for recurrent+MTP graphs.
- Build boundary fixtures at 15/16/17 and higher views through the real Meta context mechanism, plus repeated eval/reset.
- Port the minimal constant/rationale only if failure is reproducible; prefer a derived bounded formula over magic 128.
- Measure metadata memory and verify ordinary dense/non-recurrent graphs are unchanged.
- Record capacity decision and keep the change correctness-scoped, not a performance claim.

## Detailed Solution & Technical Design

Recurrent snapshot views are estimated around 2*(n_rs_seq+1) per shared recurrent layer. Headroom concerns tensor-object metadata, not model data; bound it from graph structure and retain safe failure if capacity is exceeded.

## Code Samples & Guidance



## Files

ggml-backend-meta.cpp; Meta context boundary fixture; recurrent/MTP graph construction and eval/reset tests; metadata memory evidence.

## Validation

15/16/17+ view allocation; real recurrent+MTP graph; repeated reset; metadata accounting; dense/non-MTP controls; source identity.

## Effort & Risk



## Standards

Affirmative capacity proof; bounded resource accounting; no performance claim for correctness promotion.

## Acceptance Criteria

Existing 16 limit is shown insufficient or a proven bound requires change; new bound covers declared maximum with bounded metadata cost; no non-target behavior changes.

## Notes

Supersedes: NRO10
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-nasone-rdna-optimizations-nro10

## Change Log

- 2026-09-09T10:52:50.276971+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:09:14.697625+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.090326+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.739019+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:44:03.530481+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria
- chg_20260910_024433_three-more-nasone-successors-n_7555
- 2026-09-10T02:44:33.131554+00:00 (updated-by): Updated: section:ledger-events
