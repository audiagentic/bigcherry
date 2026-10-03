---
id: RNX10
order: 10
plan: patching-r9700-cross-rdna
state: pending
created-at: '2026-10-03T01:33:42.485216+00:00'
breadth: ''
skill: advanced
created-by: codex
work: L
priority: P0
---

# R9X10 — Qwen4Exp shared-expert decode fusion

## Description

Reduce the shared-expert branch to the smallest safe launch set, independently of adopting MXFP4/FP8. Use the actual BigCherry quant/type path first.

## Steps

- Rocprof the shared-expert window at M=1 and MTP verify; count launches and bytes.
- Map shared gate/sigmoid, gate-up, SiLU/mul, down, residual/combine graph nodes to mmvq/mmq dispatch.
- Fold the scalar gate into an existing required read/quant or down epilogue without changing projection precision.
- Fuse SiLU/mul with required activation conversion, then add a skinny GEMM only if it remains material after RNX04/MMVQ work.
- Add trace marker and explicit disable path; ablate 1215 overlap ON/OFF.

## Detailed Solution & Technical Design

Analyze the remaining shared-expert tail after upstream up+gate+SwiGLU fusion: ffn_gate_inp_shexp MUL_MAT -> SIGMOID -> MUL with ffn_shexp -> ADD into moe_out. Any fusion must preserve PARTIAL semantics under tensor split and leave the following AllReduce unchanged. Coordinate with existing 1215 shared-expert stream overlap and RNX04 HC work; do not duplicate either owner. First establish HIP graph capture behavior and whether removing roughly three launches per layer changes E2E performance before authoring a package.

## Code Samples & Guidance



## Files

- kernels/r9k_moe_mxfp4a8.hip
- r9700_vllm/moe/shared.py
- r9700_vllm/kernels/moe.py
- tests/test_shared_expert_r9k.py
- src/models/qwen4exp.cpp
- ggml/src/ggml-cuda/mmvq.cu
- ggml/src/ggml-cuda/mmq.cuh
- ggml/src/ggml-cuda/ggml-cuda.cu

## Validation

Compare gate logit, sigmoid result, gate-up output, post-SiLU/mul activation, down output and final model output against the unfused graph. Cover M=1/2/4/8, MTP depths, shared-gate extremes, multiple quants, single GPU and tensor split. Profile with 1215 OFF/ON to prove the gain is not just an overlap artifact.

Acceptance: measurable E2E decode/MTP benefit on at least one declared generation, exact fallback elsewhere, and no dependency on an unpromoted R9X05 format experiment.

## Effort & Risk



## Standards

Original source alias is R9X10. Proposed slot 1310; use actual Q/IQ/Q8 path unless RNX05 independently promotes a format.

## Acceptance Criteria

- Promote only with measurable E2E decode/MTP benefit and exact fallback elsewhere.
- No dependency on an unpromoted RNX05 format experiment.
- Keep 1207 routed top-k weighting and 1215 stream overlap ownership distinct.

## Notes

Existing owners: 1207, 1215, 1237, 1265.

Verbatim legacy source retained during R9X→RNX migration:

# R9X10 — Qwen4Exp shared-expert decode fusion

Status: planned
Proposed patch: `1310_r9x_shared_expert_fusion`
Depends on: R9X01; coordinate with R9X04 router/HC and R9X05 only if a new weight format is actually selected
External source: `kernels/r9k_moe_mxfp4a8.hip`, `r9700_vllm/moe/shared.py`, `r9700_vllm/kernels/moe.py`; test `tests/test_shared_expert_r9k.py`
Existing BigCherry owners: `1207_rd17_moe_topk_down_fold`, `1215_rd394041_amd_stream_moe_overlap`, `1237_rd30_moe_mmq_compact_grid`, `1265_rd30b_moe_mmq_compact_grid_rdna4_rdna2`



Review RV4205 assessment: a possible 1310 fused `b + a * sigmoid(dot(w,x))` tail is a design candidate, but its value depends on graph-capture A/B results and exact PARTIAL ownership. GPT design pass (req_83e7cdc000be4b9e) reinforces launch-fusion-first ordering and warns against duplicate shared-expert/HC implementations. Further analysis required before code.

2026-10-04 consolidation: owner item for shared-expert launch fusion. PRBE13 (shared-expert output-chain fusion, blocked on locating the real anchor) is merged here and superseded. Keep separate: PRBE34/PRBE101 (shared expert on an auxiliary stream - concurrency, not fusion). Online (RV4214): AMD-Ecosystem Set 4 MMV epilogue fusions (sigmoid/SiLU, multiply, residual folded into MMV/MMVQ) cut 1263->1103 launches/token, +1.4% tg; upstream already fuses sigmoid*mul. Flash-Next: Q8_0 shared expert, ~4,176 launches per MTP step per GPU.

## Goal

Reduce the shared-expert branch from a chain of gate, projection, activation/quantization, down-projection and final gate launches to the smallest safe set of kernels. This is a P0 launch-overhead experiment and is **independent of adopting MXFP4/FP8**: first implement against the quant/type BigCherry actually serves.

## Exact source mapping

`r9700_vllm/moe/shared.py` replaces the shared expert with a four-stage path: row quant + expert-gate dot/sigmoid, gate-up GEMM, SiLU/mul+quant, down GEMM with the gate folded into the per-row epilogue. The source helper `r9k_quant_rows_fp8_gate` is implemented in `kernels/r9k_moe_mxfp4a8.hip`; `tests/test_shared_expert_r9k.py` is the numerical reference.

## Exact BigCherry mapping

Start at the shared-expert branch in `src/models/qwen4exp.cpp` and record the concrete ggml nodes for: shared gate projection/sigmoid, shared gate-up, SiLU/mul, shared down, and residual/combine. Then map their backend dispatch to the deployed quant path:
- `ggml/src/ggml-cuda/mmvq.cu` for MMVQ/decode paths such as existing 1241/1274 work;
- `ggml/src/ggml-cuda/mmq.cuh` for MMQ/MUL_MAT_ID-style paths such as 1237/1265;
- `ggml/src/ggml-cuda/ggml-cuda.cu` only for dispatch/fusion registration if existing fusion metadata cannot express the epilogue.

Do not fuse routed-expert top-k weighting owned by 1207 into this patch. Do not conflate 1215's **stream overlap** of routed/shared branches with this patch's **within-shared-branch launch fusion**; both may coexist and must be ablated together.

## Implementation sequence

1. Rocprof the exact shared-expert window on Qwen4Exp for M=1 and MTP verify; count launches and bytes.
2. First fold the shared-expert scalar gate into an already-required read/quant kernel or down epilogue without changing projection precision.
3. Next fuse SiLU/mul with the activation conversion/quantization already required by the downstream projection.
4. Only write a custom skinny GEMM if the remaining generic projection launch is still material after R9X04/other MMVQ work.
5. Add `BIGCHERRY_PATCH_HIT patch=1310_r9x_shared_expert_fusion` and an explicit disable path.

## RDNA adaptation

- RDNA4: source scheduling is reference, but use BigCherry's actual Q/IQ/Q8 weight path unless R9X05 independently promotes a new format.
- RDNA3: expected high-value because pointwise fusion and eliminated intermediate traffic are architecture-neutral; retune vector width/occupancy.
- RDNA2: prioritize gate/activation/epilogue fusion while retaining baseline GEMM if a custom skinny kernel loses. No FP8/MXFP4 dependency is allowed in the common fusion contract.

## Change Log

- 2026-10-03T01:33:42.485216+00:00 (created-by): Created by codex
- 2026-10-03T01:39:27.319095+00:00 (updated-by): Updated: section:notes

## Reviews

- RV4205
- 2026-10-03T02:23:22.204124+00:00 (updated-by): Updated: section:detailed_solution, section:notes
- 2026-10-03T15:21:48.339442+00:00 (updated-by): Updated: section:notes
