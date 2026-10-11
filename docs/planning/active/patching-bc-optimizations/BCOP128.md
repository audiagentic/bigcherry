---
id: BCOP128
order: 128
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-11T02:05:52+00:00'
created-by: agent
priority: P2
work: S
---

# MET09: classify MoE numerical divergence before changing routing

## Discovery / disposition

Pinned b11474 Qwen4Exp uses score-only descending `ggml_argsort_top_k` for MoE routing and accumulates weighted experts in selected-slot order. Exact ties and F32 nonassociativity are *possible* but unproven causes of historical row/expert split TV 0.055–0.113 and MTP acceptance differences. The original MET09 assertion that patch 1281 disables MMQ broadcast dedup is stale: QFP30 already added conditional range dedup and a -1-safe scatter. Patch 1281 SUMMARY also described only Phase A despite existing GPU qualification in its README. This run corrected those contradictions; no hardware gain is claimed.

## Ownership, novelty and exclusions

MET09 owns first-divergence classification; MET02/1281 owns range kernels; MET08 owns usage-placement performance; MET11 owns load-time permutation; QFP36 owns router tuning. Last independent MET09 commit 2026-10-06T14:53:18Z, and no 12-hour independent MET09/1281 commits, PRs or queued evidence were found in the inspected selected/active branches. Recent BCOP119–127/previous blocked BCOP128 addressed other mechanisms. QFP49/1360/1361, QFP50/1359, QFP41/1356 and QFP17/1330 were protected; none were changed. No second sorter, cache, scheduler, telemetry or dispatch table.

## Bounded action / terminal gate

Reuse existing fidelity D/S/D2 and Qwen4Exp graph callbacks to record the **first divergent token/layer**: pre-argsort scores, kth boundary gap, inverse-mapped selected-ID set and order, weights, per-expert outputs, actual MMVQ/MMQ/dedup route, and weighted sum. If membership differs without exact ties, reject a tie-break and route to router numeric analysis; if only order differs, compare host canonical-order F32 sums without changing production; if same IDs/order but outputs differ, hand a reproducible kernel fixture to MET02. If no divergence exceeds D/D2/CPU reference floor, close as numerical-tolerance behavior. No hardware queue or implementation before the protected prefill work terminates.

## Evidence / validation

Pinned `src/llama-graph.cpp::build_moe_ffn`, `ggml/src/ggml.c::ggml_argsort_top_k`, `ggml/src/ggml-cuda/{argsort.cu,mmq.cu,ggml-cuda.cu}`, `1281/patch.py` and README. Source-static assertions 11/11 passed. Disposable host bitonic fixture: 720/720 unique-score permutations invariant, 480/720 tied-score permutations changed top-2 membership, 600/720 changed order; F32 sum outcomes 0/1/2; three range-sentinel fixtures passed. Synthetic, **not** GPU proof. [llama.cpp #30262](https://github.com/ggml-org/llama.cpp/pull/30262) closed unmerged; [#29998](https://github.com/ggml-org/llama.cpp/pull/29998) open Vulkan-only; [vLLM route capture](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/routed_experts_capturer.py) is an observability reference. No repository pytest, C++/HIP build, hardware run or new benchmark performed.
