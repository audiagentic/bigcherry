---
id: PNRO15
order: 0
plan: patching-nasone-rdna-optimizations
state: pending
created-at: '2026-09-09T10:53:17.647208+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: P2
---

# Dequant-float MMVDQ for Q4_K/Q5_K/Q6_K decode

## Description

Evaluate dequant-float MMVDQ for Q4_K/Q5_K/Q6_K decode as a separate algorithm family, beginning with plain matvec before fused SwiGLU.

## Steps

- Cross-reference AMD PR #61 and nasone 670512... ancestry; avoid duplicate kernel families.
- Extract only MMVDQ; exclude nasone graph-optimization defaults.
- Implement explicit opt-in Q4_K/Q5_K/Q6_K kernels for ne11==1 contiguous/non-batched layouts with GGML_CUDA_DQ_MMV/DQ_Q6K/DQ_ROWS controls.
- Keep ordinary MMVQ as fallback; validate dequant math against canonical reference on adversarial blocks/scales and ncols=1.
- Measure removed Q8_1 launch/time against added float math/memory; sweep rows-per-block and architecture separately.
- Evaluate dense fused SwiGLU only after plain MMVDQ correctness and performance are established.

## Detailed Solution & Technical Design

MMVDQ trades activation quantization for float activation/dequant work; assess against F32 reference and model quality, not only native MMVQ. It is not an MMVQ geometry candidate and must retain architecture-specific opt-in.

## Code Samples & Guidance



## Files

MMVDQ HIP/CUDA kernels; quant-format dequant fixtures; selector/env controls; plain matvec tests; optional fused GLU child; call-weighted decode campaign.

## Validation

Q4_K/Q5_K/Q6_K reference and block edges; ncols=1; nonqualifying dense/MoE fallback; fused/unfused GLU; architecture controls; Q8 launch accounting; quality and decode timing.

## Effort & Risk



## Standards

Cross-source deduplication; plain MMVDQ before fused GLU; explicit opt-in/architecture selector; quality gate.

## Acceptance Criteria

All three formats meet tolerance; nonqualifying layouts fall back; at least one target architecture has repeatable decode gain without quality regression; graph-opt defaults remain outside.

## Notes

Supersedes: NRO16
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-nasone-rdna-optimizations-nro16

## Change Log

- 2026-09-09T10:53:17.647208+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:09:49.336023+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.116971+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.779729+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:46:16.757889+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria
- chg_20260910_024630_the-remaining-nasone-successor_5195
- 2026-09-10T02:46:30.083528+00:00 (updated-by): Updated: section:ledger-events
