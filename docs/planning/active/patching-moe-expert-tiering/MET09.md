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

# Explain expert-split, row-split and expert-order numerical divergence

## Decision and measured baseline (2026-10-11)

**Pending a causal discriminator; do not implement a stable top-k, MMQ or accumulation patch from the historical correlation alone.** Historical BigCherry next-token probes reported expert-vs-row split TV 0.055–0.113 and top-1 agreement 19–23/24; a partition change altered MTP acceptance approximately 55% to 77%. With `GGML_CUDA_ALLREDUCE=none`, the compared MD5s and TV 0.1132 were unchanged, excluding compressed AllReduce as the cause in that measured lane. MET08 separately recorded 22/24 top-1 and TV 0.080 for a usage-placed layout; those are distinct experiments and must not be pooled. These are measured *differences*, not an identified mechanism or a new speedup.

## Current implementation trace and corrected premise

1. Pinned llama.cpp b11474 `src/llama-graph.cpp::llm_graph_context::build_moe_ffn` selects experts via `ggml_argsort_top_k(ctx0, selection_probs, n_expert_used)` (~2160), gathers weights with `ggml_get_rows` (~2174), computes expert outputs and reduces them in selected **slot order** through `ggml_view_2d` / sequential `ggml_add` (~2375–2400). This is **not** QSA's `ggml_top_k`: `ggml/src/ggml.c::ggml_argsort_top_k` is a descending `ggml_argsort` followed by a view.
2. `ggml/src/ggml-cuda/argsort.cu::bitonic_step` compares only F32 scores, without an original expert-ID secondary key. CPU `ggml/src/ggml-cpu/ops.cpp::cmp_argsort` likewise compares scores only. Optional CUB/hipCUB selection must be read from the *compiled* build. Exact ties **can** change top-k membership under expert permutation; finite near-ties can change if router arithmetic changes. Neither was demonstrated in the real divergent tokens.
3. BigCherry `1281_moe_mul_mat_id_range/patch.py` **no longer disables broadcast dedup unconditionally**. Its QFP30 range path sets `bc_range_dedup = bc_range && !use_native_fp4 && ne11 == 1 && n_expert_used > 1`, translates global IDs to local or -1, initializes the inverse map to -1, and dispatches `quantize_scatter_range_mmq_q8_1_cuda` for the qualifying MMQ branch. This supersedes the original MET09 claim that all range nodes bypass `dedup_bcast`. Native FP4, nonbroadcast, MMVQ and fallback cases are distinct. A route through MMVQ is selected by `ggml_cuda_mul_mat_id`'s architecture/type/batch thresholds, **not guaranteed** by setting `-b 1 -ub 1` alone.
4. `1281/README.md` records validated CPU/GPU range-op cases and actual EP end-to-end results. Its `SUMMARY.md` retained an obsolete Phase-A/CPU-only description, corrected alongside this plan. No GPU experiment was run during this audit.

**Cheapest host discriminators actually executed:** an exact score-only descending bitonic-network emulator preserved the original top-2 order for 720/720 permutations of six distinct scores; with three tied top scores, 480/720 permutations changed top-2 *membership* and 600/720 changed ordered top-2. Four F32 contributions produced sums 0, 1 and 2 under different addition orders. Three global-to-local/-1 range fixtures passed. These are synthetic possibility proofs, not evidence that production router logits contain ties. Source-static assertions: 11/11 passed.

## One bounded attribution procedure (existing owners and telemetry)

**Gate A — acquire one correlated first-divergence record, not a new open-ended hardware campaign.** Reuse `tools/lab/flash-next/flash-fidelity.sh`, its D/S/D2 controls and existing Qwen4Exp graph callbacks (`ffn_moe_argsort`, `ffn_moe_topk`, `ffn_moe_weights`, `ffn_moe_down`, `ffn_moe_out`). Compare the *same original prompt/token/context* on original GGUF row split, reordered GGUF row split, and expert split, with MTP disabled for the attribution baseline; then repeat with the production MTP depth. Capture only the first divergent token/layer and its adjacent layer to bound storage. Existing artifacts must include build/pin/recipe, GGUF hashes, permutation manifest/hash, ROCm/device order, `-ts`, batch/ubatch, quantization, graph state, seed and actual output token. Do not modify or queue QFP49/1360/1361 or other protected lanes.

Capture per selected layer: original expert IDs (inverse-map reordered indices), F32 router logits and selection scores **before** argsort, kth/(k+1)th score gap and ULPs, ordered top-k IDs, ordered weights, per-expert F32 output *before* weighted sum, sum order and final output; identify the backend/rank that computed each stage. Capture exact MMVQ/MMQ kernel names, `bc_range_dedup` activation, activation quantization, and byte/row counts. An absent artifact or a one-time process marker is **not** proof of a specific per-token kernel route. Check D-vs-D2 repeatability before attributing S-vs-D.

**Gate B — classify without changing production arithmetic:**

| Observed first divergence | Cheapest next test | Disposition |
| --- | --- | --- |
| Selected global-ID **set** differs and exact kth boundary tie is proven | Host stable ordering by **original** expert ID; check original/reordered route identity, CPU/HIP reference and repeated requests | Consider a default-off tie-policy change only if model semantics and real-output correctness require it. Never infer a tie from small TV alone. |
| Selected set differs but kth boundary has a positive gap | Compare pre-argsort F32 router logits, quantized gate weights and per-device computation | Route-numeric cause; a tie-break is rejected. Return to router owner if a true correctness defect is established. |
| Same selected set but different **slot order** | Host-only canonical-order F32 accumulation of identical per-expert outputs; compare bit patterns | Summation-order hypothesis only. Sorting selected IDs in production changes the baseline and requires separate explicit equivalence policy. |
| Same IDs and order, different per-expert outputs | Same-token `MUL_MAT_ID` reference with range on/off, exact kernel/quant route and weights; compare F32 logits/KLD | MET02/1281 owns any confirmed range-kernel defect. Do not assume dedup is off. |
| Same MoE output; later logits/text differ | Trace downstream first divergence | Close MET09's MoE cause and hand off to the actual owner. |

**Gate C — correctness and terminal gate.** Cover equal/near-equal/NaN/Inf scores, score permutation, duplicate-ID *negative controls*, M=1/2/4/8, quant Q8_0/IQ4_XS, MMVQ/MMQ, original/reordered GGUF, single gfx1100/gfx1201 and mixed no-P2P tensor split, repeated same-process requests, multi-ubatch, long context and MTP acceptance. Assert expected-vs-observed expert work and transfer bytes; do not call missing work a speedup. Compare greedy first divergence, logits/KLD and TV against D/D2 and optional CPU-F32 reference. **Close with a numerical-tolerance explanation** if deviations are within the measured reference/repeat floor; **fail closed** and assign a concrete MET02 or router defect if out-of-envelope and reproducible. A new performance patch requires a separately proven exposed E2E cost and correctness before paired ABBA benchmarks; no throughput improvement is claimed here.

## Upstream and other engines (inspected 2026-10-11)

- [llama.cpp #30262](https://github.com/ggml-org/llama.cpp/pull/30262) (created 2026-10-10; **closed unmerged**) proposed fixing duplicate expert IDs in CUDA `mm_ids_helper` by counting lower experts once per token. Its code/diff is useful for a hostile duplicate-ID reference test, but ordinary `ggml_argsort_top_k` selects unique original indices, so duplicate IDs are **not** an established cause of MET09's real runs. Do not port unmerged code or claim AMD qualification.
- [#29998](https://github.com/ggml-org/llama.cpp/pull/29998) (open; updated 2026-10-10) handles duplicate expert rows in Vulkan through a compact prepass; [#29822](https://github.com/ggml-org/llama.cpp/pull/29822) is closed unmerged. Neither is a HIP range-MMQ or deterministic tie-break fix.
- [#28713](https://github.com/ggml-org/llama.cpp/pull/28713) merged 2026-10-08 and changes `GGML_OP_TOP_K` algorithm selection, **not** the Qwen4Exp MoE `ggml_argsort_top_k` contract.
- [vLLM RoutedExpertsCapturer](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/routed_experts_capturer.py) captures actual per-layer `topk_ids` for replay; use the **observability mechanism**, not its CUDA kernels or routing policy. [SGLang MoE top-k consolidation](https://github.com/sgl-project/sglang/issues/26771) warns against multiplying model-specific top-k kernels; neither engine supplies transferable RDNA performance or an agreed tie policy.

## Ownership / no duplicate implementations

MET09 owns only the **numerical attribution and terminal disposition**. MET02/1281 owns range semantics, dedup scatter and MMQ/MMVQ correctness; MET08 owns usage-placed layout and performance; MET11 owns load-time permutation; QFP36/1357 owns router tuning; QFP49 owns currently active prefill/transfer work. Reuse existing `flash-fidelity.sh` and graph callbacks. Do not create another router, sorter, dispatch table, telemetry registry, expert cache, scheduler or patch until Gate B identifies a proven defect and its owner. The selected capability had no independent plan/patch/PR/queued work within 12 hours; last independent MET09 commit was 2026-10-06T14:53:18Z. No protected paths changed.

## Change log

- 2026-10-06T14:53:14.519034+00:00 — initial plan and measured split/order differences.
- 2026-10-11 — source audit corrected the obsolete range-dedup assumption, separated MoE argsort from QSA TOP_K, and bounded first-divergence classification; see BCOP128.
