---
id: NRO16
order: 16
plan: nasone-rdna-optimizations
state: superseded
created-at: '2026-09-08T09:50:40+10:00'
breadth: ''
skill: advanced
created-by: agent
priority: P2
work: L
---

# Dequant-float MMVDQ for Q4_K/Q5_K/Q6_K decode

## Description

Retain nasone commit `670512936d83d12dd41d27c8b30e4bb416a47f31` as a viable deferred candidate, not a discarded idea. The source adds a decode matvec family that dequantizes K-quant weights to float and dots directly with the original F32 activation, avoiding the Q8_1 activation-quantization step used by MMVQ. It covers Q4_K/Q5_K/Q6_K single-column contiguous dense matvec and includes a fused dense SwiGLU variant.

BigCherry's AMD registry already records the related AMD MMVDQ source as `excluded` only because it was too large for the first follow-up slice; that is not negative performance/correctness evidence. This NRO item cross-links the newer integrated implementation and reframes disposition as deferred viable.

## Steps

1. Cross-reference AMD PR #61 tracked commit and determine code/content ancestry versus nasone `670512...`; avoid duplicate implementations of the same kernel family.
2. Extract MMVDQ only. The nasone commit also defaults graph optimization on RDNA3.5; that policy is out of scope and must not ride along.
3. Add Q4_K/Q5_K/Q6_K kernels with F32 activation, exact eligibility (`ne11==1`, contiguous/non-batched), and runtime opt-out.
4. Keep ordinary MMVQ as fallback/control.
5. Validate dequant math against canonical CPU/backend reference over adversarial blocks/scales.
6. Measure removed Q8_1 quantization launch/time versus added float math and memory traffic.
7. Sweep rows-per-block 1/2/4/8 and architecture separately; do not inherit RDNA3.5 default to gfx1100 without evidence.
8. Evaluate dense fused SwiGLU as a child/secondary arm after plain MMVDQ is correct.

## Detailed Solution & Technical Design

MMVDQ trades arithmetic/weight-dequant work for removal of activation quantization and potentially simpler accumulation. Decode is often bandwidth/launch dominated, so the trade can win even with more FP math. It is a new algorithm family relative to existing MMVQ candidate geometry and should not be encoded as merely an MMVQ nwarps/VDR variant.

Accuracy changes because activation is no longer quantized to Q8_1; this may improve or alter results. Compare to F32 reference and model quality, not only native MMVQ output.

## Code Samples & Guidance

Source env controls: `GGML_CUDA_DQ_MMV`, `GGML_CUDA_DQ_Q6K`, `GGML_CUDA_DQ_ROWS`. Preserve explicit opt-in on architectures without validated default.

## Files

Planning-only in initial NRO landing; future large kernel package and dedicated backend-op fixtures.

## Validation

Backend reference across quant types/block edge values, ncols=1 boundary, dense/MoE non-selection, fused/unfused GLU, real decode call-weighted timing.

## Effort & Risk

Very high code surface; multiple quant formats and new kernels. Worth retaining because it targets a known decode overhead class.

## Standards

Cross-source deduplication, atomic plain-MMVDQ before fused GLU, architecture-specific selector, quality gate.

## Acceptance Criteria

- Correct dequant-float outputs within pre-registered tolerance for all three quant types.
- Safe fallback for all nonqualifying layouts/batches.
- Repeatable decode gain on at least one target architecture with no quality regression.
- Graph-opt default remains outside this experiment.

## Notes

This item changes the *disposition semantics* from 'not first follow-up' to 'deferred viable'; it does not erase the old registry history.

Superseded by: PNRO15
Migration: capability-rebaseline-v3-2026-09

## Change Log

- 2026-09-08T09:50:40+10:00 (created-by): Created from current nasone MMVDQ integration and prior AMD tracking; P2.

## Ledger-events


- Pending: ag-ledger MCP unavailable in authoring session.
- 2026-09-09T11:25:41.048079+00:00 (updated-by): Updated: section:notes
- 2026-09-09T11:43:47.270167+00:00 (state-transition): State: pending → superseded
- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:57:59.922609+00:00 (updated-by): Updated: section:ledger-events
