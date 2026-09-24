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

TODO. Grepped b11126 for "mmvdq"/"DQ_MMV": zero matches -- confirmed not upstream. Verified the existing MMVQ kernel family this item must sit alongside: `ggml/src/ggml-cuda/mmvq.cu` implements quantized-activation matvec via `mul_mat_vec_q<type, ncols_dst, has_fusion, small_k, halve_iters>` (~line 601), with per-type dispatch helpers `get_vec_dot_q_cuda()` (~line 40), `get_vdr_mmvq()` (~line 69), `get_device_table_id()` (~line 106/124), `calc_nwarps()` (~line 452), `calc_rows_per_block()` (~line 579), and launch wrapper `mul_mat_vec_q_switch_fusion` (~line 1011). MMVDQ (dequant-float matvec: dequantize each block to FP32 then do float matvec, vs MMVQ's Q8_1-quantize-the-activation approach) is a distinct algorithm family per the item's own framing and sibling patch 1204_rd08_q6k_mmvq_vdr2 (existing MMVQ VDR2 tuning) confirms mmvq.cu is the right neighborhood. Genuinely TODO.

## Steps

- Cross-reference AMD PR #61 and nasone 670512... ancestry; avoid duplicate kernel families.
- Extract only MMVDQ; exclude nasone graph-optimization defaults.
- Implement explicit opt-in Q4_K/Q5_K/Q6_K kernels for ne11==1 contiguous/non-batched layouts with GGML_CUDA_DQ_MMV/DQ_Q6K/DQ_ROWS controls.
- Keep ordinary MMVQ as fallback; validate dequant math against canonical reference on adversarial blocks/scales and ncols=1.
- Measure removed Q8_1 launch/time against added float math/memory; sweep rows-per-block and architecture separately.
- Evaluate dense fused SwiGLU only after plain MMVDQ correctness and performance are established.

## Detailed Solution & Technical Design

MMVDQ trades activation quantization for float activation/dequant work; assess against F32 reference and model quality, not only native MMVQ. It is not an MMVQ geometry candidate and must retain architecture-specific opt-in.

Add new device kernel(s) `mul_mat_vec_dq<type, ncols_dst>` alongside (not replacing) `mul_mat_vec_q` in mmvq.cu or a new sibling file mmvdq.cu included from the same dispatch point. Each kernel: for ne11==1 (single-column, non-batched) contiguous layouts only -- dequantize the K/Q4_K/Q5_K/Q6_K block to FP32 in-kernel using the existing `dequantize_kernel_t`/type-trait dequant functions already used by ggml-cuda's dequant paths (`ggml/src/ggml-cuda/dequantize.cuh` or similar -- NEEDS-VERIFICATION of exact header/function names), then accumulate FP32 matvec directly against the FP32 (or BF16) activation instead of quantizing the activation into Q8_1 first as MMVQ does. Gate selection with three build/env controls named in the item: GGML_CUDA_DQ_MMV (master opt-in switch), GGML_CUDA_DQ_Q6K (Q6_K-specific enable, since Q6_K has different block layout/cost), GGML_CUDA_DQ_ROWS (rows-per-block tuning knob mirroring calc_rows_per_block's role for MMVQ). Selection order in the host dispatch: if opt-in enabled AND ne11==1 AND contiguous AND type in {Q4_K,Q5_K,Q6_K} AND architecture allow-list matches, call MMVDQ; else fall through to the existing `mul_mat_vec_q_switch_fusion` (ordinary MMVQ), which remains the unconditional default.

## Code Samples & Guidance

Target file: `ggml/src/ggml-cuda/mmvq.cu` (b11126). Verified anchor: line 601 `static __global__ void mul_mat_vec_q(` -- new kernel(s) inserted after the closing brace of this template (insert after the existing mul_mat_vec_q definition block, before line ~851 `mul_mat_vec_q_moe`). Host dispatch anchor: line 1011 `static void mul_mat_vec_q_switch_fusion(` -- the new MMVDQ opt-in branch wraps this call site (mode="insert_before" for a routing check that redirects to the new kernel launcher when the DQ env/build gates + shape/type/ne11==1 conditions hold, otherwise falls through unchanged to the existing switch_fusion call).
NEEDS-VERIFICATION before authoring the patch.py: exact dequant header/function names in ggml-cuda for per-block-to-float conversion of Q4_K/Q5_K/Q6_K (grep `ggml/src/ggml-cuda/*.cuh` for `dequantize_block` or similar), and the exact host function signature at the mul_mat_vec_q_switch_fusion call site (its caller, to find where the opt-in redirect must actually live).
Patch package sketch: `patches/1263_pnro15_mmvdq_dequant_float/{patch.toml,patch.py,SUMMARY.md,README.md,TESTING.md}` (next free id; adjust if 1262 is claimed by PNRO13 and 1262/1263 by others -- use next actually-free slot at authoring time). patch.toml: kind="enhancement", state="untested", requires=[].

## Files

MMVDQ HIP/CUDA kernels; quant-format dequant fixtures; selector/env controls; plain matvec tests; optional fused GLU child; call-weighted decode campaign.

ggml/src/ggml-cuda/mmvq.cu (or new mmvdq.cu); dequant helper headers (TBD, needs-verification); patches/1263_pnro15_mmvdq_dequant_float/*; Q4_K/Q5_K/Q6_K reference/block-edge fixtures; architecture-gated decode campaign evidence.

## Validation

Q4_K/Q5_K/Q6_K reference and block edges; ncols=1; nonqualifying dense/MoE fallback; fused/unfused GLU; architecture controls; Q8 launch accounting; quality and decode timing.

Offline: `PYTHONPATH=tools python -m bigcherry patch-lint patches/1263_pnro15_mmvdq_dequant_float`; `PYTHONPATH=tools python -m bigcherry patch-rebase-check --focal-overlay 1263_pnro15_mmvdq_dequant_float --source bigcherry-tuning`; test-backend-ops-style fixtures: Q4_K/Q5_K/Q6_K reference match at adversarial blocks/scales and ncols=1, nonqualifying dense/MoE layouts must fall back to ordinary MMVQ unchanged. Hardware (Brutus, gfx1100/gfx1201, not run here): `python -m bigcherry.patch.validation_campaign --overlay 1263_pnro15_mmvdq_dequant_float --arch gfx1100` measuring removed-Q8_1-launch vs added-float-math tradeoff, call-weighted decode timing per architecture, plus model quality gate (not just native MMVQ comparison); promote per-architecture independently.

## Effort & Risk

Work=L. Kernel-correctness risk is the dominant hazard (silent numerical drift from a dequant bug is easy to miss without adversarial-block fixtures); architecture-specific opt-in keeps blast radius small. NEEDS-VERIFICATION items (dequant header names, exact switch_fusion call site) must be resolved by reading real source before patch.py is authored -- do not paste unverified anchors.

## Standards

Cross-source deduplication; plain MMVDQ before fused GLU; explicit opt-in/architecture selector; quality gate.

## Acceptance Criteria

All three formats meet tolerance; nonqualifying layouts fall back; at least one target architecture has repeatable decode gain without quality regression; graph-opt defaults remain outside.

## Notes

Supersedes: NRO16
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-nasone-rdna-optimizations-nro16

2026-09-24 relevance at b11126: TODO. Grounded directly against b11126 mmvq.cu source (function names/line numbers verified by grep); GPT design request submitted (dev-gpt-agent, batched with PNRO13/PNRO14) but the gateway queue was saturated and did not return within this session's budget -- design completed directly from source instead.

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
- 2026-09-24T02:30:42.710721+00:00 (updated-by): Updated: section:description, section:detailed_solution, section:code_samples, section:files, section:validation
- 2026-09-24T02:31:05.751575+00:00 (updated-by): Updated: section:effort_risk, section:notes
