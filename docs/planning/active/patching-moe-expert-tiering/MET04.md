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
3. Sweep cache budgets at 0/5/10/20% of host-resident expert bytes on gfx1100/gfx1201, and include an explicit 4096 MiB point because the Shali12/Stew-derived R9700 report found a large decode gain there. Test small decode batches, MTP depth 3, and pp512/2048/8192 to expose any large-batch bypass or resident-layer displacement.
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

Add a provenance reproduction lane on gfx1201 using the closest compatible Flash-Next artifact/configuration to the Shali12 report: `-ncmoe 41`, explicit 0 versus 4096 MiB expert-cache budget, speculation/MTP off for the primary cache attribution, then MTP on as a separate interaction test. Match `--lazy-mode on --load-mode none` where the tested implementation exposes equivalent controls. Run at least two requests in the same process with the gather path both at its production setting and, where applicable, the Stew r30 control setting. Record all deviations from the source configuration.

## Effort & Risk

Medium for cache qualification; high for EP. Risks: HIP support, PCIe miss latency, cache thrash under MTP/multi-request workloads, and sacrificing resident layers/prefill throughput to cache VRAM.

## Standards

Reuse-before-fork. One placement owner. Compare at equal VRAM budget. Separate decode/prefill promotion gates. Do not infer AMD performance from CUDA results. No EP implementation until cache/static-tier evidence shows an unresolved bottleneck. Preserve source lineage for externally derived mechanisms and benchmark hypotheses.

## Acceptance Criteria

- Upstream-style cache path proven supported or unsupported on HIP gfx1100/gfx1201.
- Equal-VRAM static-tier/cache A/B includes TG, MTP, pp512/2048/8192, hit rate, upload bytes/time and VRAM.
- The Stew/Shali R9700 4096 MiB cache result is either reproduced directionally on local gfx1201 or dispositioned with measured reasons; source URL, source revision/configuration and local deviations are retained in evidence.
- Multi-request correctness explicitly covers the reported `GGML_SCHED_DEVGATHER` second-request corruption class; first-request-only success cannot promote the cache path.
- Cache promotion requires repeated decode improvement and <=1% regression in every regime where enabled; otherwise dispatch is regime-specific.
- EP proceeds only if cache/static tiering leaves a measured service/transfer bottleneck worth its complexity.
- EP uses MET01 profile data and reports per-device service balance plus aggregation cost.
- No duplicate placement registry, router semantics, expert store or cache plan.

## Notes

With 10 active experts/token EP has exploitable parallelism, but temporal locality may avoid host compute/transfer on most accesses with much less scheduler complexity. Cache qualification therefore precedes EP.

2026-10-05, folded in from BCOP17 (audit backfill) - cache before expert parallelism: compare static tiering with upstream-style GPU expert caching (llama.cpp #29887, not in pin `050439614`) at equal VRAM; sweep cache budgets 0/5/10/20%, cold/warm and forced miss/hit controls, TG plus pp512/2048/8192, upload/eviction/displaced-layer telemetry; proceed to EP only if a quantified bottleneck remains. MET05 auxiliary-device execution stays separate.

2026-10-06 provenance audit: connected the existing cache-before-EP plan to its practical AMD lineage in `stew675/llama-cpp-rdna-boosts` and the single-R9700 reproduction notes in `Shali12/r9700-flash-next-notes`. Added the reported 4096 MiB cache point, `-ncmoe 41`, lazy/no-load configuration, MTP interaction, and second-request DEV_GATHER corruption as explicit reproduction/correctness gates. These remain external evidence until reproduced locally.

References:
- https://github.com/ggml-org/llama.cpp/pull/29887
- https://github.com/ggml-org/llama.cpp/pull/29943
- https://github.com/stew675/llama-cpp-rdna-boosts
- https://github.com/Shali12/r9700-flash-next-notes

2026-10-06 HOP PROBE (owner question: experts on the R9700 with the main model on the XTX cards) - REJECTED FOR THE CURRENT TOPOLOGY. Runs moecopy-b11402h-r9700-hop3 and -hop3-mtp (clean, sequential; build b-moe-expert-cache-b11402h; Flash-Next UD-IQ4_XS, ctx 32768, f16 KV, every expert in VRAM, no patch needed - placement by -ot). T = production tensor split over 2x XTX + R9700 (flashnext profile); Ln = -sm layer over the two XTX with the routed experts of the last n layers on the R9700 (activations go XTX -> R9700 -> XTX through host memory, no P2P). 28 layers do not fit (33 GiB of experts on the 32 GB card), so n = 24 and 14. Without MTP: T decode 40.9 / 40.9 t/s, 23.8K-token prefill 1335 / 1321 t/s; L24 decode 24.4 t/s, prefill 927 t/s; L14 decode 25.1 t/s, prefill 1024 t/s. With the MTP sidecar on the 6900 XT: T decode 86.5 / 86.2 t/s, prefill 1011 / 1059 t/s; L24 decode 56.3 t/s, prefill 820 t/s; L14 decode 60.3 t/s, prefill 887 t/s. So the layer-split-plus-expert-device layout costs ~40% of decode without MTP (~30-35% with) and 23-30% of prefill against the tensor split, and putting more expert layers on the R9700 makes it slightly worse (L24 < L14). Greedy text is identical between L24 and L14 and differs from T (different split, expected). Note what this does and does not attribute: the arms differ in the whole execution model (three cards computing each layer in parallel under the tensor split vs one card at a time under the layer split) as well as in the expert hop, so this rejects 'dense on XTX / experts on the R9700 via layer split' as a replacement for the production split; it does not measure whole-expert parallelism inside a tensor split (1283's actual design). MTP on the 6900 XT works unchanged in every one of these layouts.

## External provenance and traceability

This cache-before-EP direction has two distinct source lineages and they must remain distinguishable in reports:

- **Stew implementation lineage:** `https://github.com/stew675/llama-cpp-rdna-boosts`. This is an RDNA/HIP llama.cpp patch set and is the provenance for the practical AMD expert-cache mechanism being investigated. Treat Stew-specific behavior as mechanism evidence, not as an upstream contract.
- **R9700 reproduction/report lineage:** `https://github.com/Shali12/r9700-flash-next-notes`. On a single Radeon AI PRO R9700/gfx1201 (32 GB VRAM, 64 GB RAM, ROCm 10), the report used Flash-Next with `--lazy-mode on --load-mode none`, `-ncmoe 41` and `MOE_EXPERT_CACHE_MIB=4096`, reporting roughly 17-20 t/s without the cache and 36-39 t/s with the 4 GiB cache. Its MTP draft head reportedly reached roughly 52-55 t/s while consuming another ~3-8 GB VRAM. These are external observations to reproduce, not BigCherry measurements.
- **Correctness provenance from the same R9700 report:** with the r30 MoE path, leaving `GGML_SCHED_DEVGATHER` enabled could produce apparently correct first-request output followed by corrupted `////////` output on the second request; `GGML_SCHED_DEVGATHER=0` was the reported workaround and r31 changed the default. This is the reason MET cache qualification requires at least two requests in one process and must not accept first-request-only correctness.
- **Upstream convergence lineage:** llama.cpp #29887 is the upstream-style GPU cache for host-resident MoE experts and #29943 moves selective expert copying behind a user-code scheduler copy callback. BigCherry should reproduce the Stew/R9700 result as evidence, but prefer the upstream seam when it provides the required semantics and performance.

Every benchmark or implementation note derived from these sources must record source URL plus tested commit/revision when available. Do not relabel external Stew/Shali results as local BigCherry results. If a mechanism is reimplemented through #29887/#29943 rather than Stew code, preserve the lineage as `idea/evidence -> Stew/R9700 report; implementation seam -> upstream llama.cpp`.

## Change Log

- 2026-10-02T04:45:00.827445+00:00 (created-by): Created by agent
- 2026-10-05: BCOP17 audit backfill added cache-before-EP gate.
- 2026-10-05: Transplanted equal-VRAM cache qualification, telemetry, ownership and EP decision gates from `automation-qfp-indexer-20261004`.
- 2026-10-06: Added Stew/Shali source traceability and converted the R9700 4 GiB cache/MTP/multi-request observations into explicit qualification gates without creating a second cache owner.
- 2026-10-06T01:04:53.986663+00:00 (updated-by): Updated: section:notes


## 2026-10-06 topology/cache follow-up

The completed R9700 hop probe closes the layer-split workaround: do not place expert-bearing layers on the PCIe-x4 R9700 as a separate stage to avoid host-cache traffic. It loses about 30-40% decode and 23-30% prefill versus production tensor split in the measured lanes. This is a terminal rejection for that layout, not evidence against 1283 whole-expert parallelism inside tensor split.

MET01 now owns the remaining cache/residency question. Its next discriminator is static-hot versus equal-VRAM resident layers versus 1337 LRU under the measured ~6.7 GB/s R9700 H2D ceiling. MET04 must not create another cache policy, placement solver or hop layout. Reopen EP only after MET01 reports a residual that cannot be solved by residency/cache policy and transfer accounting shows expert parallelism can reduce critical-path bytes rather than add host-bounce traffic.
