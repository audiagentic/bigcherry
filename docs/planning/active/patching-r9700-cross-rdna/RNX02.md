---
id: RNX02
order: 2
plan: patching-r9700-cross-rdna
state: pending
created-at: '2026-10-03T01:32:50.179535+00:00'
breadth: ''
skill: advanced
created-by: codex
work: M
priority: P1
---

# R9X02 — Decode attention and QSA/indexer extraction

## Description

Extract the r9700-stack head-dim-256 decode/paged-attention and sparse QSA/indexer ideas for Qwen4Exp without duplicating existing FlashAttention tile tuning. Dense attention and QSA score/top-k remain separate acceptance units unless profiling proves fusion is required.

## Steps

- Review 1270, 1202, 1203, 1266, and 1271 before touching FlashAttention.
- Map Qwen4Exp graph construction to exact CUDA/HIP score, top-k/index-gather, and sparse-attention operations.
- Profile D=256 decode and QSA independently, including shapes, types, KV layout, token counts, MTP path, launch count, and bytes.
- Implement 1300 only as exact-shape/capability dispatch around the existing attention path; implement 1301 only for an independently measured QSA hot path.
- Keep KV numerical-format experiments independently bisectable.

## Detailed Solution & Technical Design

Create one default-off package-only experiment at ggml/src/ggml-cuda/fattn.cu::ggml_cuda_get_best_fattn_kernel(). When Q has D=256 and one query token, K/V are Q8_0, the existing D=256 Q8 vector case is available, and the architecture is RDNA3/RDNA4, return the existing BEST_FATTN_KERNEL_VEC path under an explicit opt-in flag. Do not add a kernel or dequantizer. Preserve the current selector for all unmatched cases and leave gfx103x unchanged initially.

## Code Samples & Guidance



## Files

- kernels/r9k_attn.hip
- kernels/r9k_qsa.hip
- kernels/r9k_qsa_score.hip
- r9700_vllm/kernels/attn.py
- r9700_vllm/attn/qsa.py
- src/models/qwen4exp.cpp
- ggml/src/ggml-cuda/fattn-tile.cuh

- patches/1300_rnx02_q8kv_vec_decode/patch.toml
- patches/1300_rnx02_q8kv_vec_decode/patch.py
- patches/1300_rnx02_q8kv_vec_decode/SUMMARY.md
- tools/tests/patch/test_1300_rnx02_q8kv_vec_decode.py

## Validation

Mechanics: exact positive and negative selector predicates, including non-Q8 types, non-D=256, multi-token Q, unavailable vector case, and non-RDNA3/4. Later hardware validation must cover n_tokens={1,2,4}, GQA variants, context lengths around padding/256 boundaries, backend-reference parity, graph replay, greedy/KLD checks, and unprofiled 80K E2E decode on gfx1100/gfx1201. Promotion requires measured E2E crossover evidence, not a kernel microbenchmark alone.

## Effort & Risk



## Standards

RDNA4 may use source geometry after semantic parity; RDNA3 retunes existing gfx11 primitives; RDNA2 ports only architecture-neutral ideas without gfx12 WMMA.

## Acceptance Criteria

The experiment is default-off and activated only by an explicit opt-in; it selects the pre-existing Q8 D=256 vector implementation only for the exact supported predicate; no new kernel, dequantizer, or staging path is introduced; unmatched selector behavior is unchanged; numerical and end-to-end evidence is recorded before any lifecycle promotion.

## Notes

Original source alias is R9X02. Proposed slots 1300 and 1301; 1201/1202/1203/1266/1270/1271 remain overlap owners.

Verbatim legacy source retained during R9X→RNX migration:

# R9X02 — Decode attention and QSA/indexer extraction

Status: planned
Proposed patches: `1300_r9x_attn_d256_decode`, `1301_r9x_qsa_score_indexer`
Depends on: R9X01
External source: `kernels/r9k_attn.hip`, `kernels/r9k_qsa.hip`, `kernels/r9k_qsa_score.hip`, `r9700_vllm/kernels/attn.py`, `r9700_vllm/attn/qsa.py`
Existing BigCherry owners: 1201, 1202, 1203, 1266, 1270, 1271



GPT design pass (req_83e7cdc000be4b9e): existing Q8_0/D=256 vector attention is already instantiated in fattn-vec.cuh, while the competing MMA/TILE path can stage the full K/V tensors. Recommended advisory package: 1300_rnx02_q8kv_vec_decode. This is a dispatch experiment, not a validation claim; quantify both staging removal and numerical/E2E tradeoffs.



Implementation update (2026-10-03): created advisory package 1300_rnx02_q8kv_vec_decode. It is untested, default-off, and deliberately not added to a recipe. The package targets the verified c061df198 fattn.cu selector anchor and uses the existing Q8_0/D=256 vector implementation.



Review RV4206 assessment: the QSA path has a separate cross-start greedy determinism defect in tied radix TOP_K selection; keep that as a distinct 1301 analysis/repair and do not fold it into 1300. The 1300 experiment remains justified by the measured verify-width q8_0->f16 attention cost, but requires backend parity and deterministic output checks.

## Goal

Extract the r9700-stack head-dim-256 decode/paged-attention and sparse QSA/indexer ideas that reduce launch/memory cost in Qwen4Exp, without duplicating BigCherry's existing FlashAttention tile tuning. Treat dense attention and QSA score/top-k as separate acceptance units unless profiling proves they must be fused together.

## BigCherry/llama.cpp mapping

Review `patches/1270_pnro14_rdna35_fa_tile_d256/patch.py` first: its exact target is `ggml/src/ggml-cuda/fattn-tile.cuh` and it already owns a gfx1151 D=256/ncols=32 tile row. Also review `1202_rd04_bf16_flash_attn_tile`, `1203_rd050607_rdna4_wmma_fa_q6k_mmq`, `1266_rd05_wmma_fa_tileq_sync`, and `1271_prbe54_q5_kv_dequant_f16` before touching FlashAttention.

For the QSA/indexer path, map graph construction in `src/models/qwen4exp.cpp` to the exact CUDA/HIP ops that implement score calculation, top-k/index gather, and sparse attention. If the operation lowers through generic `mul_mat`/top-k rather than a named QSA backend file, patch the narrow dispatch/fusion point; do not introduce a new op solely to mirror vLLM naming.

## Implementation sequence

1. Profile D=256 decode and QSA score/index selection independently. Capture shapes, types, KV layout, token counts, and whether MTP verify takes the same path.
2. Diff r9k attention/QSA algorithms against current `fattn-*` implementation. Extract only novel dataflow: paged-KV access pattern, score staging, reduction/top-k fusion, fewer materialized intermediates, or launch-count reduction.
3. `1300`: add an exact-shape/capability dispatch around the existing attention implementation. Keep generic/tile-table behavior unchanged outside the predicate.
4. `1301`: only if QSA has an independently measurable hot path, add the sparse score/indexer optimization with a standalone disable/trace path.
5. Never fold a KV numerical-format experiment into these patches; 1271 and KV quantization work must remain independently bisectable.

## RDNA adaptation

| Generation | Plan |
|---|---|
| RDNA4/gfx12 | Use r9k geometry/instruction sequence as the reference, after parity with llama.cpp semantics. |
| RDNA3/gfx11 | Reuse paging/staging/fusion. Retune block/wave layout; use existing gfx11 FA primitives instead of gfx12-only WMMA. Test gfx1100 and gfx115x separately where dispatch differs. |
| RDNA2/gfx103x | Port only memory-layout/launch-reduction ideas that survive without gfx12 WMMA. Prefer existing RDNA2 attention math primitive and wave configuration. |

## Change Log

- 2026-10-03T01:32:50.179535+00:00 (created-by): Created by codex
- 2026-10-03T01:38:28.533176+00:00 (updated-by): Updated: section:notes
- 2026-10-03T02:18:04.994678+00:00 (updated-by): Updated: section:detailed_solution, section:validation, section:acceptance_criteria, section:notes

## Reviews

- RV4206
- 2026-10-03T02:21:03.292673+00:00 (updated-by): Updated: section:files, section:notes
- RV4213

## Ledger-events




- chg_20261003_022112_added-a-safe-opt-in-q8-attent_5492
- 2026-10-03T02:21:15.702342+00:00 (updated-by): Updated: section:ledger-events
- 2026-10-03T02:23:15.628294+00:00 (updated-by): Updated: section:notes
- chg_20261003_040732_long-context-qwen4exp-output-i_9401
- 2026-10-03T04:07:36.135523+00:00 (updated-by): Updated: section:ledger-events
- chg_20261003_084248_investigated-and-rejected-a-fl_8872
- 2026-10-03T08:42:51.875082+00:00 (updated-by): Updated: section:ledger-events
- chg_20261003_104706_faster-long-context-decoding-f_1057
- 2026-10-03T10:47:17.743572+00:00 (updated-by): Updated: section:ledger-events
