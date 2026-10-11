---
id: MET09
order: 9
plan: patching-moe-expert-tiering
state: pending
created-at: '2026-10-06T14:53:14.519034+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P2
work: S
---

# Explain MoE expert-order and split numerical differences (source-first attribution)

## Authoritative b11552 audit — 2026-10-12 (supersedes the original Steps and cause ranking)

**Disposition:** pending only for a bounded first-divergence attribution. No new tie-break, routing kernel, MMQ activation packer or benchmark queue. MET09 owns diagnosis; MET02/1281 owns range-MUL_MAT_ID arithmetic; MET04/1283 owns expert split; MET08 owns usage-placed performance; MET11 owns future load-time expert permutation. This item is not an implementation owner.

### Existing measured facts (historical, not reproduced in this audit)

- Expert split versus row split: next-token TV **0.055–0.113**, top-1 **19–23/24**; MTP acceptance and greedy text can differ. The same row-split configuration on original versus physically reordered GGUF can produce different text. Those are observations, **not proof of a tie, routing bug, or invalid result**.
- The historical `GGML_CUDA_ALLREDUCE=none` control retained identical MD5s and TV 0.1132, arguing against compressed collective wire as the sole cause in that run. It does not exclude arithmetic ordering or a changed routing path.
- MET08's 2026-10-07 128/128/256 EP layout fitted ctx 245760 with 1341; prefill was approximately equal and decode improved by a few percent alongside changed acceptance. Do not attribute the decode gain solely to faster kernels without matched accepted/drafted-token counts.

### Verified code and corrected assumptions

1. **Router is ARGSORT, not QSA TOP_K.** `src/llama-graph.cpp::build_moe_ffn` calls `ggml_argsort_top_k(selection_probs, n_expert_used)`; `ggml/src/ggml.c::ggml_argsort_top_k` creates `GGML_OP_ARGSORT` descending plus a view. In b11552 `ggml/src/ggml-cuda/argsort.cu::bitonic_step` compares scores with strict `>`/`<` and has no explicit original-expert-ID tie key. `src/models/qwen4exp.cpp` separately uses `ggml_top_k` for QSA attention pools. **Patch 1294 modifies `ggml/src/ggml-cuda/top-k.cu` (QSA TOP_K), not MoE ARGSORT.** Do not extend or copy 1294 merely to fix MoE routing.
2. **A fused MoE router may bypass ARGSORT.** `ggml/src/ggml-cuda/ggml-cuda.cu::ggml_cuda_topk_moe_fusion` and `ggml_cuda_should_use_topk_moe` may dispatch `ggml_cuda_op_topk_moe` instead of the ordinary graph chain. `ggml/src/ggml-cuda/topk-moe.cu::topk_moe_cuda` explicitly resolves equal scores to the **lower current physical expert index** in its warp reduction (also with bias). Thus even a deterministic fused tie rule is **not invariant under a physical expert permutation**. Record the actual executed route before changing an ARGSORT kernel. The b11552 and inspected current upstream implementations retain this distinction.
3. **The old range-MMQ premise is stale.** `1281_moe_mul_mat_id_range/patch.py` now has QFP30's `bc_range_dedup` and `quantize_scatter_range_mmq_q8_1_cuda`. For `bc_range && !use_native_fp4 && ne11==1 && n_expert_used>1`, range nodes **do reuse broadcast activation dedup**, with -1 inactive inverse-map slots. Native FP4 and non-broadcast cases still differ. The old blanket assertion that 1281 disables `dedup_bcast` is false on this branch. Source admission is not evidence that the range-dedup kernel executed on the production model.
4. **Different arithmetic trees remain possible.** `1283_qwen4exp_expert_parallel/patch.py` changes Meta expert-axis split state to partial outputs and range MUL_MAT_ID per device, followed by a reduction; ordinary row split partitions expert down-projection rows. A different F32 accumulation order can alter low bits even with identical selected original experts and weights. The historical TV does not quantify how much each mechanism contributes.
5. **Placement contract:** `tools/lab/flash-next/expert-place.py` uses `order[layer][new_id]=old_id` and permutes all expert weight tensors plus router rows (B1). On equal scores at the selection cutoff, a physical-index tie-break can select different **old** experts. MET11's B2 design retains original router rows/scores, selects old IDs, then maps `old_id -> new_id` **only for expert weight dispatch**, preserving top-k slot order and its associated router weights. A global sort of selected IDs without permuting the paired weights is incorrect. The B2 map must remain graph-owned and stable across same-process requests and replay.

### Cheapest discriminator and precise decision tree

**Gate 0 — host/static, completed:** pinned source inspection found distinct `ARGSORT`, fused `topk_moe`, and QSA `TOP_K` routes, plus the QFP30 range-dedup addition. A disposable float32/permutation fixture passed three permutations with tied and unique scores, one non-associative F32 sum witness, and four range-dedup admission cases. It proves only logical possibilities, not the production root cause. No build or GPU execution occurred.

**Gate 1 — one trace, not another open-ended campaign:** using existing `ffn_moe_argsort`, `ffn_moe_topk`, `ffn_moe_weights` graph callback names and 1338 routing/profile infrastructure, add a *temporary* bounded dump at the **first divergent layer/token** for matched original and reordered GGUF arms. Capture pin/patch set, layer, context, ubatch, MTP off, split, actual fused-vs-ARGSORT route, full F32 router score bits around the cutoff, top-k slot index, physical ID, inverse-mapped original ID, router weight bits, selected expert count, and downstream output/logit hashes. The aggregate STRP frequency profile alone cannot establish per-token ties. Reuse existing trace paths; remove disposable logging after attribution.

**Gate 2 — classify and terminate:**
- Different inverse-mapped original IDs with **bit-equal** router scores and an exact cutoff tie: attribute to physical-index tie policy. Validate B2 old-ID selection + post-selection translation in MET11; do **not** create a second global top-k implementation. A deterministic physical-ID tie-break alone does not solve permutation invariance.
- Different original IDs without a bit-equal tie: compare original-ID-aligned router score bits and fused/ARGSORT route first. Treat altered scores, quantization or route as the cause candidate; do not label it a tie bug.
- Identical original IDs/slot weights but differing expert outputs: compare `MUL_MAT_ID` dispatch identity (MMVQ vs MMQ, Q8_1 range-dedup marker, native FP4), per-rank partials, F32 summation order and logits; use a single-token MTP-off control **only after confirming actual MMVQ execution**. `-b 1 -ub 1` alone is not dispatch proof.
- No first divergence under matched controls: terminate this hypothesis; do not queue further work. No reproduction => no new patch.

### Validation and promotion fences

Require same original GGUF bytes/model revision, quant/KV, context, graph and ubatch, stable rank mapping and physical device order, identical seeds and greedy decoding, MTP-off primary oracle and MTP-on secondary interaction. Compare per-layer original IDs and paired weights exactly; for numerical drift record full-vocabulary logits/KLD, top-1, TV against a repeat-arm floor and CPU/F32 control where feasible. For split changes require expected-versus-observed expert work and partial-sum/transfer accounting, repeated same-process requests, long context and graph replay. A faster run with fewer selected experts is invalid. Do not treat the historical 0.146 TV envelope as proof of equivalence without matched controls.

**Terminal disposition:** no observed route/tie mismatch => retire tie proposal; QFP30 range dedup already covers the admitted Q8_1 path => retire duplicate dedup proposal; only an isolated reproducible discrepancy with a matching reference may justify an owner-scoped implementation. No performance promotion without correctness and measured end-to-end gain on gfx1100/gfx1201; gfx1030 is a negative/reference control, not a P2P participant.

### External mechanisms / sources

- Pinned llama.cpp b11552: [router](https://github.com/ggml-org/llama.cpp/blob/b11552/src/llama-graph.cpp), [ARGSORT](https://github.com/ggml-org/llama.cpp/blob/b11552/ggml/src/ggml-cuda/argsort.cu), [fused MoE top-k](https://github.com/ggml-org/llama.cpp/blob/b11552/ggml/src/ggml-cuda/topk-moe.cu). Upstream source is a correctness/dispatch baseline, not a measured AMD gain.
- [vLLM expert-map manager](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/expert_map_manager.py) distinguishes global/local/physical expert IDs and unsupported kernels; use its mapping contract as a reference, not its CUDA/EP performance. [SGLang modular MoE](https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/expert_parallelism.mdx) similarly separates routing, dispatch and combine; no direct port or new BigCherry subsystem.

## Historical notes (superseded)

The 2026-10-06 draft inferred that 1281 disabled broadcast dedup and proposed a new expert-ID tie-break. The current patch already restores range dedup in eligible Q8_1 cases, and the fused router already breaks ties by **physical** index. Historical experiments remain evidence; their unverified causal ranking and open-ended implementation steps are superseded above.
