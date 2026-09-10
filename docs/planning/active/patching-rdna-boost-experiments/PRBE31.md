---
id: PRBE31
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:55:35.677451+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: null
---

# AMD-GEMM-004: Large-M F16 shadow to tuned hipBLASLt crossover

## Description

Derive the large-M crossover between native quantized MMQ and tuned hipBLASLt over a PRBE29 F16 shadow, preserving negative evidence.

## Steps

- Require PRBE29 shadow and optionally PRBE30 identity; define Qwen3.6-27B Q8_0 dense signatures with Q4/Q6 secondary.
- Sweep M=64,128,192,256,384,512,768,1024,2048,4096 and compare native MMQ, default hipBLASLt and tuned hipBLASLt.
- Use M<=128, decode and memory-constrained controls; validate tolerant/exact output and PPL first.
- Record per-architecture/per-quant kernel/E2E PP, VRAM and load cost; derive crossover from measurements, not fixed threshold.
- If tuned MMQ always wins, reject shadow dispatch while retaining EC10-style negative evidence.

## Detailed Solution & Technical Design

Capability owner: patching

Split assessment: One independent boundary; Build/Run support is a dependency.

Overlap assessment: No duplicate boundary found; related items are prerequisites or adjacent evidence.

## Code Samples & Guidance



## Files

PRBE29 shadow and optional PRBE30; dense MUL_MAT dispatch; hipBLASLt/native controls; M-sweep campaign; correctness/PPL/VRAM/load artifacts; EC10 disposition.

## Validation

Output/PPL; M sweep; native/default/tuned controls; per-arch/quant kernel+PP; VRAM/load; decode/memory controls; durable crossover or negative result.

## Effort & Risk



## Standards

Derived threshold; dependency-aware; preserve negative result; no shadow default from kernel-only timing.

## Acceptance Criteria

Only a measured per-architecture/quant crossover with positive E2E benefit promotes; if no crossover exists, explicitly reject dispatch and retain evidence.

## Notes

Supersedes: RD38
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd38

## Change Log

- 2026-09-09T10:55:35.677451+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:12:47.175446+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.267648+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.009190+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:57:05.654512+00:00 (updated-by): Updated: section:description, section:steps, section:files, section:validation, section:standards, section:acceptance_criteria
- chg_20260910_025719_dense-gemm-successors-prbe293_6872
- 2026-09-10T02:57:19.708858+00:00 (updated-by): Updated: section:ledger-events
