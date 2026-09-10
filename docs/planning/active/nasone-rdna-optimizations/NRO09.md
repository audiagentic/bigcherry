---
id: NRO09
order: 9
plan: nasone-rdna-optimizations
state: pending
created-at: '2026-09-08T09:50:40+10:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: L
---

# GPU-resident LRU cache for host-offloaded MoE expert weights

## Description

Evaluate nasone commit `6ed7fb04f902bbfffe536eb0b46cd2d3b87b8a28`, which adds a per-layer GPU-resident LRU cache for expert slices when MoE weights are intentionally resident in host RAM. The source workload (Qwen3.8-Flash-Next, 512 experts/10 routed, many expert layers offloaded) observed substantial short-term routing locality despite near-uniform long-run use, making a bounded hot-expert cache potentially valuable.

This is a capacity/bandwidth optimization, not a replacement for fully resident MoE kernels. It should trigger only for host-resident expert layers and single-token decode; prefill/batch must retain the stock graph initially.

## Steps

1. Freeze source commit and enumerate every ABI/graph modification: CLI/context parameters, routing observation callback, cache-layer ownership, device tables, CPU skip table, GPU cached `mul_mat_id` chain, async upload worker, and decode-boundary publication.
2. Prove exact algebra: CPU path zeroes cached expert rows, GPU cache path maps uncached IDs to a guaranteed zero slot, and the two outputs sum to the original result.
3. Separate cache policy (LRU, slot count, insert throttle) from graph correctness. First validate a static mapping before enabling asynchronous replacement.
4. Validate cache-slot lifetime and publication: never expose a new id->slot mapping until the full expert slice upload completed.
5. Test eviction while requests are active, request/context teardown, repeated model/context creation, and no callback use-after-free.
6. Instrument hit/miss/insert/evict rates and bytes avoided; source-reported hit rate is not BigCherry evidence.
7. Sweep slot count and insert throttle under multiple routing traces, including adversarial no-locality and phase-changing traces.
8. Measure host-memory traffic, PCIe traffic, decode TPS, cache VRAM footprint, and time-to-warm.
9. Keep fully GPU-resident models as a strict non-selection control.
10. Promote only for host-offloaded MoE configurations where net throughput/VRAM tradeoff is proven.

## Detailed Solution & Technical Design

The source creates device companion tensors containing K cached expert slices plus an all-zero dummy slot. A device id->slot table remaps routed IDs for the cached GPU `mul_mat_id` chain. A host table supplied to the CPU path causes cached experts to be skipped and their output rows zeroed. The outputs are summed, preserving exact decomposition if mappings are coherent.

The hard problem is asynchronous cache maintenance. Routing is observed after a graph, candidate inserts are selected, copies occur on a worker/stream, and the mapping changes only at a safe decode boundary after upload completion. BigCherry should model mapping generations explicitly in diagnostics so stale/torn-map bugs are detectable.

## Code Samples & Guidance

Initial experiment should use a small deterministic cache and a synthetic routing trace before real async LRU. Do not optimize the callback/worker until exact row ownership is proven.

## Files

Future package after implementation review; this item is planning-only in the initial NRO landing. Likely surfaces: common args, ggml CPU `mul_mat_id`, llama context/graph, new cache support folded or added according to patcher capabilities, and targeted tests.

## Validation

Reference output for every routing ID; cached/uncached split; dummy slot; repeated eviction; asynchronous publication; thread teardown; prefill non-selection; fully resident non-selection. Performance must include warmup and cache memory cost.

## Effort & Risk

Very high. Cross-backend graph decomposition, host/device mapping coherence, asynchronous worker lifetime, and significant new VRAM allocation. A stale mapping can return a different expert's weights without crashing.

## Standards

Correctness-first, bounded resource accounting, explicit source SHA, no global callback lifetime leaks, decode-only scope until separately expanded.

## Acceptance Criteria

- Exact result versus stock graph for deterministic mapping and dynamic LRU stress.
- No torn/stale slot publication under repeated eviction.
- Prefill and fully resident models do not select cache path.
- Host traffic reduction is directly measured.
- End-to-end gain outweighs cache maintenance and declared VRAM budget.

## Notes

Source reported LRU-64 ~67% routing hit rate and warm decode uplift on 2x3090; these are hypotheses only and hardware/topology differs from BigCherry.

## Change Log

- 2026-09-08T09:50:40+10:00 (created-by): Created from nasone MoE expert-cache commit; P1.

## Ledger-events

- Pending: ag-ledger MCP unavailable in authoring session.
