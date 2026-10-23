---
id: PGC16
order: 0
plan: patching-gpu-collectives
state: superseded
created-at: '2026-10-05T00:00:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: L
---

# Prefill AllReduce count reduction — RETIRED as upstream duplicate

## Description

## Terminal disposition (2026-10-13)

**Do not implement the proposed independent partial-state analyzer, pre-Meta graph rewrite or extra AllReduce-fusion pass.** The intended identity `AR(a)+AR(b)=AR(a+b)` is already implemented in llama.cpp Meta at the plan's original `0504396` pin, b11552 and current upstream. PGC16 is complete as a consolidation/retirement decision, not a measured speedup. The superseded 2026-10-05 design remains accessible in Git history.

## Actual implementation and ownership

- `ggml/src/ggml-backend-meta.cpp::ggml_backend_meta_graph_compute`, `get_i_delayed_branch` and `get_i_delayed` (b11552 ~2057-2213), already combine two compatible PARTIAL branches before one ADD and delay their AllReduce boundary. The matcher requires single consumers, an immediate ADD of the two branches, same shape/type, a MIRRORED sum, compatible per-device COMPUTE flags and no intervening view/use hazard. The branch helper also recognizes safe ADD_ID, MIRRORED-MUL and view/ADD chains.
- The subgraph builder advances `i = i_delayed` (~2215-2255), so the earlier PARTIAL boundary does not cause another AllReduce. Zero-size shard COMPUTE flags are propagated to delayed nodes. `backend_ctx->comm_allreduce` and `allreduce_fallback` (~2478-2496) remain unchanged; this is boundary placement, not a new transport.
- `ggml_backend_meta_get_split_state` / `handle_bin_bcast` (~492-563, 1183-1196) already own split-state semantics. The original PGC16 `bc_partial_contract` duplicates this authority; before Meta buffer assignment its proposed buffer-derived split-state check is not generally valid.
- Native folding is conditional: a different group, extra consumer, differing type/shape, nonlinear join, host view or zero-shard flag mismatch may correctly prevent it. **No actual missed eligible fold has been demonstrated in BigCherry.**

PGC15 owns token-tiled compute/AllReduce overlap; PGC12 phase/boundary evidence; PGC14 RCCL transport; PGC09/PGC11 provider and wire precision; QFP17 prefill acceptance. Do not edit these owners or touch current Radiance helper, Meta 1340 or queued hardware work as part of this retired item.

## Upstream and other-engine comparison

- Original code: https://github.com/ggml-org/llama.cpp/blob/0504396/ggml/src/ggml-backend-meta.cpp
- Pinned code: https://github.com/ggml-org/llama.cpp/blob/b11552/ggml/src/ggml-backend-meta.cpp
- Upstream PR https://github.com/ggml-org/llama.cpp/pull/30217 merged 2026-10-09 and adds host-view handling in Meta, not the algebraic fold; the fold predates it.
- SGLang's `post_experts_sum_is_one_all_reduce` checks TP/EP group, quantized transport, shared experts, LoRA and completed outputs: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/utils.py . Use as negative correctness dimensions only.
- vLLM's AllReduce+RMSNorm fusion is post-collective and its Hopper/Blackwell results do not transfer to AMD: https://github.com/vllm-project/vllm/blob/main/docs/design/fusions.md . Do not port a second scheduler or collective.

## Evidence and bounded reopening gate

Source-static check: **9/9 native fold/guard markers** on each of `0504396`, b11552 and master. Disposable source-shaped Python host fixture: **12/12 admission/rejection cases**, plus real-arithmetic identity; `py_compile` passed. No repository pytest, C++/HIP build, GPU execution, or new measured inference performance. Existing PGC15 evidence: approximately 0.93 s compute and 2.4 s collective service for one pp2048 profile, without a count of native folds. A hypothetical uniform-cost saving of 1/128 of 2.4 s is 18.75 ms / 3.33 s (~0.56%); **illustration only, not a measurement or claim of uniform costs**.

Reopen **only** with a same-pin, same-model graph/trace receipt showing two compatible PARTIAL branches still produce two actual collectives despite native `get_i_delayed`, the exact rejecting guard and a measured exposed critical-path ceiling >=3% E2E. Reuse existing Meta/rocprof receipts and `tools/lab/flash-next/ar-segment.py`; do not queue a speculative hardware campaign. If proven, change only the existing Meta matcher behind an opt-in experiment, not the model graph builder or a new partial-state system. Require source-shaped negative fixtures, all-rank correctness including zero shards, same-process multi-request/ubatch, long-context graph replay, full logits/KLD/greedy/MTP acceptance, unchanged work accounting, then four independent paired hardware sessions with CI95-low >=3% E2E and <=1% decode regression. Otherwise retain terminal RETIRED.

## Change Log

- 2026-10-13: BCOP181 retired duplicate implementation after exact pinned/current source inspection.
- 2026-10-05: Original proposed analyzer/rewrite design; superseded, preserved in Git history.
- 2026-10-23: state 'done' (invalid) corrected to superseded

## Steps


## Detailed Solution & Technical Design


## Code Samples & Guidance


## Files


## Validation


## Effort & Risk


## Standards


## Acceptance Criteria


## Notes

