---
id: MET04
order: 4
plan: patching-moe-expert-tiering
state: pending
created-at: '2026-10-02T04:45:00.827445+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P2
work: L
---

# 1283 qwen4exp_expert_parallel: whole-expert EP across ROCm0/1/2 + CPU tail

## Description

Replace the tensor-split hot tier with whole experts per device only if lower-complexity alternatives leave a measured bottleneck. Dense/attention/GDN/HC/router/shared expert remain on the main tensor split; expert-tier branches compute locally and compact partials aggregate before the tensor-split graph resumes.

Before implementing EP, qualify the upstream-style GPU LRU cache for host-resident MoE experts (llama.cpp #29887 or its current equivalent) as a lower-complexity alternative for the same capacity problem. MET01 decides residency intent; MET04 owns runtime cache-vs-EP policy. Do not create another placement registry, router, expert store, or cache plan.

## Steps

1. Establish MET03 as correctness/performance control at identical GGUF, context and placement.
2. Qualify the upstream cache path without EP changes. Prove HIP/ROCm support explicitly; record unsupported ops/fallbacks rather than silently treating fallback as success.
3. Sweep cache budgets at 0/5/10/20% of host-resident expert bytes on gfx1100/gfx1201. Test small decode batches, MTP depth 3, and pp512/2048/8192 to expose any large-batch bypass or resident-layer displacement.
4. Instrument per cache bank: hits, misses, evictions, bytes uploaded, upload latency, expert compute latency, cache VRAM, hit rate and effective bytes/token. Hit rate alone is not a promotion metric.
5. Compare cache versus static MET03 tiering at equal total VRAM budget. If cache reaches >=95% of MET03 decode throughput with lower complexity and no >1% regression in enabled lanes, prefer the cache path and defer EP.
6. Implement whole-expert EP only if static tiering/cache leave a measured device-service or transfer bottleneck. Assign experts from MET01 profile by expected service cost, not expert count.
7. Aggregate only compact expert outputs. Measure host-staged versus available peer-copy paths before introducing any new collective mechanism.
8. Add replication only after service traces show a stable hotspot; MET01 remains placement/profile owner.

## Detailed Solution & Technical Design

The cache can remove the need for EP when routing locality is strong: repeated host-expert accesses become miss-only uploads without cross-GPU expert collectives. Conversely, low/unstable hit rate, HIP incompatibility, MTP working-set expansion, or cache VRAM displacing useful resident layers may favor static tiering or EP.

Ownership remains strict: MET01 route profiling/residency intent; MET02 range semantics; MET03 2-tier graph/CPU-tail compute; MET04 runtime host-expert cache-vs-EP decision; MET05 auxiliary-device execution. Cache keys must include expert tensor identity/layout and device. Cache policy must not alter router IDs or placement metadata.

For EP, balance predicted service cost = route frequency × measured per-expert/device service time. Expert-count balancing is insufficient on heterogeneous XTX/R9700 devices.

## Code Samples & Guidance

Telemetry per bank:

```text
{device, layer/layout_bank, cache_mib, resident_experts, hits, misses, evictions,
 hit_rate, upload_bytes, upload_ms, expert_compute_ms, cache_vram_mib,
 pp_tps, tg_tps, mtp_acceptance}
```

Keep an explicit small-batch dispatch gate and preserve normal host/offload execution above the proven crossover. If AMD crossover differs from upstream defaults, own one measured threshold here rather than forking cache semantics.

## Files

Upstream cache qualification plus BigCherry benchmark/telemetry harness. EP remains `patches/1283_qwen4exp_expert_parallel/` and Qwen4Exp tier graph/scheduler integration. Reuse MET01 placement data and MET02 range semantics.

## Validation

A/B at identical VRAM budget: MET03 static tier versus host-expert cache versus MET04 EP. Decode, MTP decode, pp512/2048/8192; gfx1100/gfx1201; cold-cache first token and warmed steady state; force 0%/100% miss cases. Capture hit/miss/eviction/upload bytes/time, per-device expert service, aggregation time, VRAM, greedy parity and KLD.

## Effort & Risk

Medium for cache qualification; high for EP. Risks: HIP support, PCIe miss latency, cache thrash under MTP/multi-request workloads, and sacrificing resident layers/prefill throughput to cache VRAM.

## Standards

Reuse-before-fork. One placement owner. Compare at equal VRAM budget. Separate decode/prefill promotion gates. Do not infer AMD performance from CUDA results. No EP implementation until cache/static-tier evidence shows an unresolved bottleneck.

## Acceptance Criteria

- Upstream-style cache path proven supported or unsupported on HIP gfx1100/gfx1201.
- Equal-VRAM static-tier/cache A/B includes TG, MTP, pp512/2048/8192, hit rate, upload bytes/time and VRAM.
- Cache promotion requires repeated decode improvement and <=1% regression in every regime where enabled; otherwise dispatch is regime-specific.
- EP proceeds only if cache/static tiering leaves a measured service/transfer bottleneck worth its complexity.
- EP uses MET01 profile data and reports per-device service balance plus aggregation cost.
- No duplicate placement registry, router semantics, expert store or cache plan.

## Notes

With 10 active experts/token EP has exploitable parallelism, but temporal locality may avoid host compute/transfer on most accesses with much less scheduler complexity. Cache qualification therefore precedes EP.

2026-10-05, folded in from BCOP17 (audit backfill) - cache before expert parallelism: compare static tiering with upstream-style GPU expert caching (llama.cpp #29887, not in pin `050439614`) at equal VRAM; sweep cache budgets 0/5/10/20%, cold/warm and forced miss/hit controls, TG plus pp512/2048/8192, upload/eviction/displaced-layer telemetry; proceed to EP only if a quantified bottleneck remains. MET05 auxiliary-device execution stays separate.

Reference: https://github.com/ggml-org/llama.cpp/pull/29887

## Change Log

- 2026-10-02T04:45:00.827445+00:00 (created-by): Created by agent
- 2026-10-05: BCOP17 audit backfill added cache-before-EP gate.
- 2026-10-05: Transplanted equal-VRAM cache qualification, telemetry, ownership and EP decision gates from `automation-qfp-indexer-20261004`.
