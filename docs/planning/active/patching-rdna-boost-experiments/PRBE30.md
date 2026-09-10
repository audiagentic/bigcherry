---
id: PRBE30
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:55:30.594236+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# AMD-GEMM-003: K-pad F16 shadow to avoid aliasing

## Description

Evaluate K-padding of the PRBE29 F16 shadow only for proven aliasing strides; dependent on PRBE29 and not an independent shadow implementation.

## Steps

- Require PRBE29 shadow identity and choose affected down_proj/other shapes by measured row-stride alias class.
- Compare padding 0 vs one cache line (and only additional values if justified) against non-alias controls.
- Validate F16 stride/consumer layout and GEMM output equivalence.
- Measure GEMM/PP, shadow memory and load/working-set overhead; keep padding conditional on aliasing stride.
- Do not combine with PRBE31 crossover in the causal arm unless explicitly declared.

## Detailed Solution & Technical Design

Capability owner: patching

Split assessment: One independent boundary; Build/Run support is a dependency.

Overlap assessment: No duplicate boundary found; related items are prerequisites or adjacent evidence.

## Code Samples & Guidance



## Files

PRBE29 shadow allocator/stride; conditional K-padding selector; affected-shape fixtures; alias/non-alias controls; GEMM/PP/memory evidence.

## Validation

Exact layout/output; aliasing and non-aliasing strides; padding boundary; GEMM/PP; memory/load overhead; PRBE29 prerequisite.

## Effort & Risk



## Standards

Dependency-aware; stride/layout correctness; conditional padding; causal isolation.

## Acceptance Criteria

Padding is selected only for proven alias classes, preserves output and yields a repeatable net benefit after memory cost; otherwise retain unpadded shadow.

## Notes

Supersedes: RD37
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd37

## Change Log

- 2026-09-09T10:55:30.594236+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:12:43.525844+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.261941+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.002185+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:56:58.392222+00:00 (updated-by): Updated: section:description, section:steps, section:files, section:validation, section:standards, section:acceptance_criteria
- chg_20260910_025719_dense-gemm-successors-prbe293_6872
- 2026-09-10T02:57:19.697607+00:00 (updated-by): Updated: section:ledger-events
