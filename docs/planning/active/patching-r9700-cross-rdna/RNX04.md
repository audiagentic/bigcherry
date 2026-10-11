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

# R9X04 — Router gate; HC POST already owned by QFP35
## Authoritative disposition — 2026-10-12, llama.cpp b11552

**Retire the proposed 1304 HC POST fusion.** It has already been implemented, validated and promoted by QFP35: production patches 1344_dsv4_hc_grid_index (dimensional PRE/POST) and 1355_hc_post_gate_fuse (the raw-injection gate). QFP35 is completed. No RNX04 HC patch, graph matcher, allocator, scheduler, extra cache, or qualification job is warranted. Historical proposal text below is retained only as superseded provenance. RNX04 remains pending solely for the independently gated 1303 router-GEMM question; 1303 is an advisory historical number, not a reserved slot.

### Exact implementation and graph/lifetime boundaries

Pinned src/models/qwen4exp.cpp::graph::build_hc_combine builds scale(inject,1/hc) -> sigmoid -> scale(2) -> ggml_dsv4_hc_post when fused_dsv4_hc_post and il>=0. The il=-1 head mixer is a different path. The native ggml/src/ggml-cuda/dsv4-hc.cu F32 HC_POST reads residual/optional comb and computes the output. 1355/patch.py inserts a four-node matcher in ggml/src/ggml-cuda/ggml-cuda.cu::ggml_cuda_try_fuse BEFORE the shorter 1313 scale/activation matcher. It checks exact hcpost->src[2] gate identity, F32 types, contiguous and same-shape intermediates, ggml_can_fuse_subgraph single-consumer safety and ggml_cuda_check_fusion_memory_ranges alias safety. It skips the three intermediate graph nodes and calls bc_cuda_op_dsv4_hc_post_gate. The kernel computes s1*sigmoid(s0*inject+b0)+b1 inside the original F32 residual/comb accumulation order. The grid is (ceil(n_embd/256),hc,n_tokens), using the existing 1344 dimensional indexing. Graph/capture, output allocation and stream ownership stay with existing ggml/HIP infrastructure; no additional H2D/D2H or GPU P2P is introduced. BIGCHERRY_HC_POST_GATE_FUSE=0 is the same-binary composed 1313/1344 control.

**Measured BigCherry evidence (b11474, not a new b11552 result):** 1355 README records pooled 8K decode 94.7 vs 94.0 t/s (+0.7%), 24K 83.8 vs 83.4 (+0.5%), and a second 24K request 64.6 vs 64.3 (+0.5%). Greedy outputs and MTP acceptance matched; prefill and non-Qwen4Exp controls did not regress. ABBA ranges overlap. 1344 separately measured +1.3–1.6% prefill; do not add these independent effects as if jointly measured. PA49's active cross-pin b11552 decode regression has not been isolated to this fusion.

**External mechanism and rejection:** bkvargyas/r9700-stack@8dfda41f kernels/r9k_hc.hip::r9k_hc_combine_norm_kernel computes BF16 combine, materializes a BF16 intermediate, then RMS normalization and affine update; r9k_hc_gate_mix_kernel reduces BF16 streams. r9700_vllm/hc.py disables these paths below 256/1024 rows respectively, so its large-prefill benefits are not transferable to llama.cpp's F32 HC_POST or 1–8 token decode. Do not port a second HC path. The fork's r9k_router_gemm_kernel is instead a potential separate comparator: BF16 input/weights, M<=32, split-K wave reductions and K divisible by 8*split. This does not establish the deployed GGUF router dtype, layout, or speed.

### Residual router gate (only remaining RNX04 ownership)

Source path: src/models/qwen4exp.cpp::build_layer_ffn -> build_moe_ffn; ggml/src/ggml-cuda/ggml-cuda.cu::ggml_cuda_mul_mat and ggml_cuda_mul_mat_cublas_impl/MMQ own physical dispatch. MET09/MET11 own expert selection/tie/permutation, QFP35/1344/1355 own HC, 1313 owns generic scale/activation, RNX09 owns integration, and currently active QFP48 owns broader decode fusion. Do not modify or queue experiments under QFP48/PA49 during their 12-hour protected window.

**Gate 0:** after those active campaigns finish, reuse an existing per-rank trace (no competing job) to identify actual router M/N/K, weight and activation type, stride, physical GPU, selected BLAS/MMQ route, MTP verify widths 1/2/4/8/16/32, kernel/transfer count, and exposed critical-rank E2E decode time. Reject/close RNX04 if even infinite router speedup saves <3% E2E, if source layout differs from a legal candidate, or if the operation is already subsumed. Do not assume global P2P on dual gfx1100 + gfx1201 with gfx1030 drafter.

**Gate 1, only if Gate 0 passes:** one disposable, default-off exact-shape router kernel using existing dispatch/fallback; source-accurate typed CPU reference for BF16/F32 rounding, split-K sum order, strides and output. Preserve original top-k/expert IDs and graph allocations. Require multi-request same-process, graph replay, full-vocab logits/KLD, greedy and MTP acceptance parity, expected/observed work and transfer accounting, and no regression for nonmatching types/architectures.

**Promotion/rejection:** at least four independent sessions per qualified architecture and ten paired ABBA rounds, matched model/context/ubatch/MTP and critical-rank timing; CI95-low >=3% E2E decode gain, <=1% control regression. Otherwise retire 1303 and close RNX04. No new benchmark is queued by this audit.

### Cheapest discriminator completed in this audit

Source-static matcher/ownership assertions 14/14; deterministic flat-index versus 3-D-coordinate mapping 61,102/61,102; host scalar semantic-distinction fixture confirmed BF16 combine+norm is not the F32 HC_POST expression. These are source/host checks, NOT compiled HIP, actual GPU results or new speedups.



## Historical description (superseded by 2026-10-12 disposition)

Reduce Qwen4Exp routing and hyper-connection decode launch/bandwidth overhead by porting dataflow rather than vLLM module registration.

## Historical steps (superseded)

- Map router logits, activation/gate, top-k, routed-expert preparation, HC up/mix, and residual combine to exact graph nodes and backend dispatch.
- Review 1279 only as an observation tool, then 1256/1257, 1237, and 1207 for existing selection and fusion ownership.
- Profile M=1/2/4/8/16/32 and derive the router cutoff from crossover data.
- Implement 1303 and 1304 as independently selectable exact-shape/architecture paths; preserve top-k tie semantics and avoid unnecessary custom ops.

## Historical design (superseded)

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

## Historical validation (superseded)

Prove intermediate and final parity against the legacy graph across M=1/2/4/8 and MTP shapes, retain CPU/fallback coverage, and census HC GEMM shapes before considering any GEMM replacement. Later hardware validation must measure launch reduction and E2E decode/prefill with no regression, coordinating with existing 1215 shared-expert stream overlap and 1207 routing/down-fold work.

## Effort & Risk



## Standards

RDNA4 source geometry is reference; RDNA3 retunes skinny-GEMM/waves; RDNA2 prioritizes pointwise/epilogue memory savings.

## Historical acceptance criteria (superseded)

The change removes only the proven gate-chain overhead; it does not duplicate generic HC fusion, changes legacy users, or bypass the CPU fallback. The custom-GEMM question remains gated on shape evidence. E2E decode and prefill show no regression before promotion.

## Notes

Original source alias R9X04. Historical 1303 is advisory only; **1304 retired** because QFP35/1355 already implements the HC POST fusion. Existing MoE owners 1207/1237/1256/1257/1279 are separate.

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
