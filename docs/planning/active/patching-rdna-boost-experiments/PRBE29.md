---
id: PRBE29
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:55:27.064093+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: null
---

# AMD-GEMM-002: Persistent F16 shadow of quantized dense weights

## Description

Evaluate persistent F16 shadows of selected quantized dense weights, with PRBE30/31 dependent and strict VRAM/workload gating.

## Steps

- Recheck current loader/device-buffer seams and define eligible dense tensors; never shadow MoE expert weights by default.
- Compare selected-tensor shadow against all-eligible shadow, with Qwen3.6-27B Q8_0 primary and Q4/Q6 economics controls.
- Validate shadow dequant contents against reference, model output/PPL, model-load overhead and VRAM delta.
- Measure PP across M/ubatch 64..4096 and TG/decode neutrality, plus VRAM-constrained and decode-only controls.
- Expose explicit opt-in and promotion only where PP gain justifies memory; publish durable identity for PRBE30/31.

## Detailed Solution & Technical Design

Capability owner: patching

Split assessment: One independent boundary; Build/Run support is a dependency.

Overlap assessment: No duplicate boundary found; related items are prerequisites or adjacent evidence.

## Code Samples & Guidance



## Files

Model load/device buffer path; selective F16 shadow allocator; dequant/reference fixtures; VRAM/load/PP/TG campaign; PRBE30/31 dependency identity.

## Validation

Shadow contents; output/PPL; load overhead; VRAM; PP M/ubatch 64..4096; TG neutrality; decode/VRAM controls; selected-vs-all tensor arms.

## Effort & Risk



## Standards

Selective shadow; explicit resource accounting; no global default; dependency-aware promotion.

## Acceptance Criteria

A declared target workload shows repeatable PP gain that justifies VRAM/load cost; no global default and no MoE shadow explosion; otherwise retain negative evidence and leave dependents blocked.

## Notes

Supersedes: RD36
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd36

## Change Log

- 2026-09-09T10:55:27.064093+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:12:39.409528+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.258047+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.995120+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:56:51.645822+00:00 (updated-by): Updated: section:description, section:steps, section:files, section:validation, section:standards, section:acceptance_criteria
- chg_20260910_025719_dense-gemm-successors-prbe293_6872
- 2026-09-10T02:57:19.678291+00:00 (updated-by): Updated: section:ledger-events
