---
id: PRBE44
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:56:32.505332+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# AMD-GDN-003: Native exp2 decay in chunked GDN

## Description

Implement and qualify native AMD exp2 decay inside the PRBE42 chunked GDN recurrence, preserving recurrent quality while reducing scalar math overhead on supported long-prefill paths.

## Steps

1. Recheck source #54 and PRBE42's decay/state representation. 2. Replace the heavier expf/OCML decay calculation with __builtin_amdgcn_exp2f or equivalent native base-2 formulation only under the exact PRBE42 hardware/shape gate. 3. Define conversion/scaling and edge-decay handling so state evolution remains within the approved tolerance. 4. Add long-sequence state/logit/PPL tests, sensitive short-sequence and edge-decay controls, and architecture fallback tests. 5. Profile instruction mix and benchmark GDN and end-to-end prefill against PRBE42 baseline.

## Detailed Solution & Technical Design

Use native AMD exp2 in the chunked GDN decay calculation to reduce scalar math overhead. Convert the decay exponent consistently with the source formulation, preserve finite/edge handling, and leave the existing expf path for unsupported architectures or numerically sensitive configurations. This is an optimization within PRBE42, not a replacement for its recurrence eligibility gate.

## Code Samples & Guidance

Trigger: long prefill sequences with the PRBE42 supported decay range. Controls: short numerically sensitive sequences, edge decay values, non-RDNA targets, and PRBE42 baseline. Boundary: input decay distribution and context length.

## Files

PRBE42 GDN decay implementation and architecture selector; state/logit tolerance and PPL tests; edge-decay/short-sequence controls; instruction-profile and PP replay evidence for AMD-GDN-003.

## Validation

Correctness: recurrent state and logits within defined tolerance against PRBE42 baseline over long sequences, with PPL/deterministic checks and edge-decay controls. Performance: instruction profile, GDN kernel time, and PP E2E with variance. Acceptance: enable only if quality remains within tolerance and kernel/E2E benefit survives; otherwise retain expf fallback.

## Effort & Risk

M; approximate exp2 can accumulate recurrent error over long contexts. Keep tolerance explicit, test edge cases, and fail closed.

## Standards

Preserve PRBE42 hardware/shape gating, recurrent-state quality, unsupported-architecture fallback, and evidence provenance.

## Acceptance Criteria

Acceptance requires long-sequence recurrent state/logit quality within defined tolerance, edge-decay and short-sequence safety, supported-architecture fallback, and a repeatable kernel or E2E prefill benefit; otherwise retain the original decay path.

## Notes

Supersedes: RD52
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd52

Supersedes RD52. Depends on PRBE42; no standalone validity outside the chunked GDN kernel.

## Change Log

- 2026-09-09T10:56:32.505332+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:13:45.997534+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.328390+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.097809+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:07:07.712309+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:effort_risk, section:standards, section:notes
- 2026-09-10T03:07:27.823252+00:00 (updated-by): Updated: section:acceptance_criteria
- chg_20260910_030747_carried-forward-the-remaining_4294
- 2026-09-10T03:07:47.473759+00:00 (updated-by): Updated: section:ledger-events
