---
id: RNX04
order: 4
plan: patching-r9700-cross-rdna
state: pending
created-at: '2026-10-03T01:33:02.882387+00:00'
breadth: ''
skill: advanced
created-by: codex
work: L
priority: P0
---

# R9X04 — MoE router and hyper-connection decode fusion

## Description

Reduce Qwen4Exp routing and hyper-connection decode launch/bandwidth overhead by porting dataflow rather than vLLM module registration.

## Steps

- Map router logits, activation/gate, top-k, routed-expert preparation, HC up/mix, and residual combine to exact graph nodes and backend dispatch.
- Review 1279 only as an observation tool, then 1256/1257, 1237, and 1207 for existing selection and fusion ownership.
- Profile M=1/2/4/8/16/32 and derive the router cutoff from crossover data.
- Implement 1303 and 1304 as independently selectable exact-shape/architecture paths; preserve top-k tie semantics and avoid unnecessary custom ops.

## Detailed Solution & Technical Design

Target the remaining HC combine chain rather than duplicating the already-fused HC PRE/POST operators. The current path is inject -> scale(1/hc) -> sigmoid -> scale(2) -> existing DSV4_HC_POST. Map src/models/qwen4exp.cpp::build_hc_combine to ggml_dsv4_hc_post in ggml/include/ggml.h and ggml/src/ggml.c, ggml/src/ggml-cuda/dsv4-hc.cu::{dsv4_hc_post_f32,ggml_cuda_op_dsv4_hc_post}, and the CPU fallback in ggml/src/ggml-cpu/ops.cpp. Add an explicitly flagged/raw-gate POST variant that computes 2*sigmoid(raw_inject/hc) inside the existing pointwise kernel, preserving legacy POST semantics and fallback behavior. Start with gfx1201/gfx1100; keep gfx103x legacy until measured. Do not replace the roughly 17,472 rocBLAS HC GEMMs with a custom low-rank kernel until an actual hc_down/hc_up/hc_inject M/N/K census justifies it.

## Code Samples & Guidance



## Files

- kernels/r9k_router.hip
- kernels/r9k_hc.hip
- r9700_vllm/router.py
- r9700_vllm/hc.py
- tests/test_router_r9k.py
- tests/test_hc_mix_r9k.py
- src/models/qwen4exp.cpp

## Validation

Prove intermediate and final parity against the legacy graph across M=1/2/4/8 and MTP shapes, retain CPU/fallback coverage, and census HC GEMM shapes before considering any GEMM replacement. Later hardware validation must measure launch reduction and E2E decode/prefill with no regression, coordinating with existing 1215 shared-expert stream overlap and 1207 routing/down-fold work.

## Effort & Risk



## Standards

RDNA4 source geometry is reference; RDNA3 retunes skinny-GEMM/waves; RDNA2 prioritizes pointwise/epilogue memory savings.

## Acceptance Criteria

The change removes only the proven gate-chain overhead; it does not duplicate generic HC fusion, changes legacy users, or bypass the CPU fallback. The custom-GEMM question remains gated on shape evidence. E2E decode and prefill show no regression before promotion.

## Notes

Original source alias is R9X04. Existing owners: 1207, 1237, 1256, 1257, 1279. Proposed slots 1303 and 1304.

Verbatim legacy source retained during R9X→RNX migration:

# R9X04 — MoE router and hyper-connection decode fusion

Status: planned
Proposed patches: `1303_r9x_router_decode_gemm`, `1304_r9x_hyper_connection_fusion`
Depends on: R9X01
External source: `kernels/r9k_router.hip`, `kernels/r9k_hc.hip`, `r9700_vllm/router.py`, `r9700_vllm/hc.py`; tests `tests/test_router_r9k.py`, `tests/test_hc_mix_r9k.py`
Existing BigCherry owners: 1207, 1237, 1256, 1257, 1279



GPT design pass (req_83e7cdc000be4b9e): HC PRE and POST are already fused; the actionable gap is the inject sigmoid/scales before POST. Recommended first step is a gated raw-inject POST microfusion, not another generic post-fusion package.

## Goal

Reduce decode launch/bandwidth overhead around Qwen4Exp routing and hyper-connection mixing. r9700-stack's router is a skinny bf16×bf16 GEMM for small M and its HC work fuses decode-side mixes/epilogues. Port the dataflow, not the vLLM module-binding mechanism.

## Mapping

Use `src/models/qwen4exp.cpp` to identify the exact graph nodes for router logits, activation/gate, top-k, routed-expert preparation, HC input/up/down/gated mix and residual combine. Use `patches/1279_moe_routing_dump` (`common/debug.cpp`) only as the routing-observation tool. Review `1256_nro07_topk_hybrid` and `1257_nro08_topk_wave32` before changing selection; review `1237_rd30_moe_mmq_compact_grid` for downstream expert launch mapping and `1207_rd17_moe_topk_down_fold` for existing fusion boundaries.

The backend target should be the existing ggml-cuda mul-mat/top-k/elementwise implementation selected by those graph nodes. R9X01 must record its exact current file/function before 1303/1304 is authored; do not add a qwen4exp-only custom op if an existing fused epilogue/dispatcher can express the same work.

## 1303 router plan

Specialize only decode-width rows where generic BLAS/MMQ launch overhead and rereads dominate. Preserve accumulation/output semantics required by the model. Measure M=1/2/4/8/16/32 and set the cutoff from crossover data, not the r9700 constant. Keep top-k as a separate stage unless fusing it produces a clear larger win and exact tie semantics can be maintained.

## 1304 HC plan

Map r9k HC up/mix and router-epilogue fusion to llama.cpp's actual HC graph. Target elimination of intermediate writes and multiple pointwise launches. Do not combine with 1303 unless one cannot be selected independently. Add shape and architecture fallbacks.

## RDNA adaptation

RDNA4: source geometry is the starting point. RDNA3: expected strong portability because the principal win is skinny-GEMM/launch reduction; retune splits/waves and use gfx11 math paths. RDNA2: prioritize fused pointwise/epilogue memory savings; retain existing GEMM if custom skinny GEMM loses on gfx103x.

## Change Log

- 2026-10-03T01:33:02.882387+00:00 (created-by): Created by codex
- 2026-10-03T01:38:41.747779+00:00 (updated-by): Updated: section:notes
- 2026-10-03T02:18:17.894951+00:00 (updated-by): Updated: section:detailed_solution, section:validation, section:acceptance_criteria, section:notes

## Reviews

- RV4204
