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
priority: P2
---

# R9X10 — Qwen4Exp shared-expert decode fusion

## Description

**RNX10 is a profiling-gated scalar shared-expert gate-tail experiment**, not another GLU, MoE GEMM, router, cache, or stream-overlap implementation. No new patch or hardware job is authorized before Gate 1. Historical patch number `1310` is **occupied by production `1310_act_q81` (QFP18)** and must not be reused.

## Implementation audit — pinned b11474 (2026-10-08)

- `src/models/qwen4exp.cpp::graph::build_layer_ffn` (1142–1189) constructs routed `build_moe_ffn`, shared `build_ffn(..., LLM_FFN_SILU, LLM_FFN_PAR)`, then shared-gate `build_lora_mm(ffn_gate_inp_shexp,cur)` -> `ggml_sigmoid` -> `ggml_mul(ffn_shexp,gate)` -> `ggml_add(moe_out,ffn_shexp)`. Gate weight `{n_embd}` yields expected gate shape `[1,M]`; shared output `[H,M]`. Confirm live graph shape, type, strides and device before counting physical kernels.
- `src/llama-graph.cpp::build_ffn` (1837–1905) already builds shared gate/up and `ggml_swiglu_split`. `ggml/src/ggml-cuda/ggml-cuda.cu::ggml_cuda_match_shared_expert` (1854–1904), `ggml_cuda_try_fuse` (3542–3559), and graph allocation-dependency registration (4653–4669) already support conditional routed+shared up/gate/GLU fusion. Do **not** duplicate it.
- `ggml_cuda.cu` (3454–3479) plus `ggml/src/ggml-cuda/unary.cu::ggml_cuda_op_unary_mul` (665–719) already fuse SIGMOID+MUL **only for same-shape operands**. `ggml_are_same_shape(other,unary)` excludes scalar-broadcast `[1,M]` vs `[H,M]` for H>1. This is static route evidence, **not proof of three actual launches**. Existing `binbcast.cu::ggml_cuda_op_mul/add` (442–451) is fallback.
- The separate `MUL_MAT+scale(+bias)` matcher (`ggml_cuda.cu` 4108–4184) requires `GGML_TYPE_NVFP4` for its scale predicate; do not presume it can fold the Q/IQ/Q8 shared gate into the down epilogue. `GGML_CUDA_DISABLE_FUSION` changes unrelated mechanisms and is diagnostic, not an isolated RNX10 A/B.

## Cheapest discriminator and implementation gate

**Gate 0, static — passed:** eight b11474 source assertions; host shapes H=2048/M=1,4,16 fail the native unary+mul equal-shape predicate while H=1/M=1 is the positive control. Existing `1310_act_q81` is registered in `config/recipes.toml`. No runtime result follows from these checks.

**Gate 1, attributed profiling — required before patching:** using existing `rocprofv3 --kernel-trace`, graph-debug and QFP18/QFP43 timing producers (no new telemetry), record `{op,shape,dtype,device,stream,kernel,bytes,elapsed}` for gate GEMV, SIGMOID, broadcast MUL, ADD and existing shared up/GLU/down. Test Qwen4Exp production quant, M=1/2/4/8 and MTP verify, pp128/512, contexts 8K/48K/98K, single gfx1100/gfx1201, dual gfx1100 tensor split and mixed gfx1100+gfx1201 no-P2P. gfx1030 only if the model fits. Record `GGML_CUDA_GRAPH_OPT`, capture/replay/reallocation, `1215+1216` ON/OFF only in its qualified single-GPU graph-opt lane, host wall, per-rank GPU busy-time **union**, H2D/D2H and physical launch counts. Do not add overlapping queue times or call total dispatch volume a tail measurement. **Close without patch** if the removable pointwise tail is <5% of E2E decode/MTP wall or exact same-device adjacency/ownership cannot be established.

**Gate 2, conditional implementation:** only after Gate 1, add an exact `GGML_OP_UNARY(SIGMOID) -> GGML_OP_MUL(broadcast) -> GGML_OP_ADD` match in existing `ggml_cuda_try_fuse` and a single F32 pointwise kernel in existing `unary.cu` or `binbcast.cu`; no new GGML op, scheduler, allocator, architecture dispatch table or quant format. Candidate computation: `dst[h,m] = moe[h,m] + shared[h,m] * sigmoid(gate[m])`. If ADD cannot safely fuse, restrict the first patch to SIGMOID+broadcast MUL. Require adjacent graph nodes, exact source identity and single consumers, `[1,M]` contiguous gate, identical `[H,M]` shared/moe/output layout, F32 dtype, supported measured M, same device/stream/rank, `ggml_can_fuse_subgraph`, `ggml_cuda_check_fusion_memory_ranges`, and allocation dependencies before graph reservation. Any unsupported layout, multi-device split, nonadjacency, graph-opt mode or alias falls back unchanged. Never introduce a cross-rank write, hidden AllReduce, host synchronization, or pointer to recycled ubatch. Use existing patch opt-in/trace/disable conventions; allocate a **fresh** patch number only after rechecking the active branch and 12-hour exclusions.

**Gate 3, correctness and promotion:** F32 host reference and backend-op cases for ±80/±0 gate logits, NaN/Inf policy, H/M and stride boundaries, both ADD operand orders, reused intermediates and negative matcher cases. Compare logits/greedy/KLD, MTP acceptance, multi-request same-process, multi-ubatch, 8K/48K/98K, graph capture/replay/reallocation, tensor-split Meta/PARTIAL zero contributions, physical transfer accounting and restart. No missing work, buffer alias corruption or new sync. Run ≥10 interleaved paired A/B rounds per supported architecture/topology against b11474 native and the production patched base; ablate 1215+1216 only on its eligible lane. **Promote only with CI95-low ≥3% E2E decode/MTP gain, ≤1% control regression, full correctness and activation**; otherwise reject and remove the experiment.

## Existing evidence, ownership and external comparison

**BigCherry measurements belong to other mechanisms:** 1215+1216 stream overlap on single gfx1100 with graph-opt ON: +2.38% mean (10 pairs, CI95 [1.45%,3.32%]); dual gfx1100 layer split ~−4.4%, tensor split neutral. 1207 routed top-k down-epilogue: −1.32% mean (6 pairs, CI95 approx [−2.38%,−0.26%]). Neither is RNX10 performance evidence. Historical ~4,176 dispatches/MTP step/GPU are not attributed to this tail.

vLLM issue #43187 proposes a single GEMV+sigmoid+broadcast scale kernel; MI355x single-run-per-cell external means +4.65% balanced/+6.36% decode-heavy/+10.29% prefill-heavy, issue closed stale 2026-09-19. SGLang `qwen2_moe.py::_append_shared_expert_ids_and_weights` and `fused_moe_triton_kernels.py::fused_append_shared_experts_with_weights(fuse_gate=True)` combine gate GEMV/sigmoid into AITER shared-expert ID append. llama.cpp GGUF does not use that append contract: **mechanism reference only**, not a direct port or RDNA3/4 benchmark. Pinned b11474 and inspected upstream master retain the existing CUDA fusion restrictions.

**Ownership:** RNX10 alone owns the scalar shared-expert tail; RNX09 owns cross-RDNA promotion. RNX04 owns HC/router, QFP35/native owns GLU, 1207/PRBE14 owns routed top-k scaling, 1215+1216/PRBE35 owns shared/routed stream overlap, and QFP18/1310 owns activation Q8_1. Do not edit protected QFP35/QFP43/MTP/patch tooling in this run. RNX10's last independent plan change was 2026-10-03 15:24 UTC; no new RNX10 patch/PR/queue was found. No build, GPU test or new hardware benchmark ran.

## Acceptance criteria

Gate 1 <5% tail wall-share or unproven ownership -> terminal no-patch disposition. Gate 1 pass -> only the bounded guarded pointwise experiment above; any correctness or CI gate failure -> reject. No speculative skinny GEMM or MXFP4 work is in RNX10 scope.


## Archived legacy proposal (non-normative; superseded above)

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
