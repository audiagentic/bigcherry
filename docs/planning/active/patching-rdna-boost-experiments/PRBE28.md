---
id: PRBE28
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:55:23.381054+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# AMD-GEMM-001: 128-byte row padding for cache-set aliasing

## Description

Qualify 128-byte row padding only for proven cache-set alias classes and architectures; preserve layout correctness and memory-cost controls.

## Steps

- Recheck current allocator/layout seams and identify float/BF16/F16 rows whose bytes are multiples of 2048 and susceptible to aliasing.
- Sweep padding 0/64/128/256 where legal, grouped by row_bytes/cache geometry; keep quantized packed rows and non-alias controls separate.
- Verify nb[1], consumers, tensor layout and model parity; capture cache counters and per-op GEMM/PP.
- Measure memory cost and reject neutral/costly padding; do not combine with dequantized shadows PRBE29–31 in the causal arm.
- Enable only through architecture/alias selector after repeatable evidence.

## Detailed Solution & Technical Design

Capability owner: patching

Split assessment: One independent boundary; Build/Run support is a dependency.

Overlap assessment: No duplicate boundary found; related items are prerequisites or adjacent evidence.

## Code Samples & Guidance



## Files

weight allocation/layout; nb[1] consumers; alias-class selector; tensor/layout/parity fixtures; cache-counter/per-op GEMM/PP campaign; separate PRBE29–31 controls.

## Validation

Float/BF16/F16 alias shapes; padding sweep; non-alias/quantized controls; tensor parity; cache counters; memory overhead; per-op and E2E PP.

## Effort & Risk



## Standards

Layout-safe; architecture/alias scoped; causal isolation; resource accounting.

## Acceptance Criteria

Padding is selected only for proven alias classes/architectures with model/layout correctness and repeatable benefit outweighing memory cost; no unconditional padding.

## Notes

Supersedes: RD35
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd35

## Change Log

- 2026-09-09T10:55:23.381054+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:12:35.177419+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.253885+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.988119+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:55:29.344874+00:00 (updated-by): Updated: section:description, section:steps, section:files, section:validation, section:standards, section:acceptance_criteria
- chg_20260910_025542_rdna-successors-prbe2628-now_5552
- 2026-09-10T02:55:42.907095+00:00 (updated-by): Updated: section:ledger-events
