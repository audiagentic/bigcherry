---
id: PRBE37
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:56:01.682197+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# AMD-FUS-001: Fuse GEMV epilogue activation

## Description

Qualify pattern-matched GEMV epilogue activation fusion as the root AMD-FUS-001 candidate, preserving fallback and PRBE38 dependency.

## Steps

- Audit current mul_mat_vec_f/q epilogues and define exact non-gated activation patterns.
- Implement only pattern-matched SILU/SIGMOID epilogue fusion; preserve unmatched graph and prefill paths.
- Validate bit/tolerance parity across Q8/Q4/Q6 and dense/hybrid decode; test graph capture and false positives.
- Measure launch count, HBM traffic and TG for 4B/9B/27B with prefill and nonmatching controls.
- Promote only if end-to-end gain is repeatable and PRBE38 consumes the exact root identity.

## Detailed Solution & Technical Design

Capability owner: patching

Split assessment: One independent boundary; Build/Run support is a dependency.

Overlap assessment: No duplicate boundary found; related items are prerequisites or adjacent evidence.

## Code Samples & Guidance



## Files

mul_mat_vec_f/q epilogue; pattern matcher/fallback; Q8/Q4/Q6 fixtures; graph capture; launch/HBM/TG campaign; PRBE38 dependency identity.

## Validation

Exact activation patterns; bit/tolerance output; graph capture; prefill/nonmatch controls; launch count; traffic; TG across model sizes.

## Effort & Risk



## Standards

Pattern-matched only; correctness before launch reduction; preserve fallback; no broad epilogue fusion.

## Acceptance Criteria

Eligible paths fuse correctly and show repeatable TG benefit with non-target paths unchanged; unsupported patterns fall back; PRBE38 depends on this validated root.

## Notes

Supersedes: RD45
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd45

## Change Log

- 2026-09-09T10:56:01.682197+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:13:15.050723+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.296126+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.050932+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:59:52.821708+00:00 (updated-by): Updated: section:description, section:steps, section:files, section:validation, section:standards, section:acceptance_criteria
- chg_20260910_030005_amd-streamfus-successors-prbe_8761
- 2026-09-10T03:00:05.781141+00:00 (updated-by): Updated: section:ledger-events
