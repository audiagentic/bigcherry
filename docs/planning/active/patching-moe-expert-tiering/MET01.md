---
id: MET01
order: 1
plan: patching-moe-expert-tiering
state: pending
created-at: '2026-10-02T04:44:44.458623+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P1
work: M
---

# Per-layer routed-expert profile + placement solver tooling

## Description

Prerequisite for every tiering patch. Collect per-layer routed-expert statistics for qwen4exp (Flash-Next) from ffn_moe_topk and turn them into a validated placement JSON under a measured per-device VRAM budget.

Expert residency has one canonical policy surface. Static hot placement, DwarfStar-style persistence/prefetch, upstream llama.cpp #29887 demand LRU, and MET05/1328's auxiliary ROCm3 residency are policies/tier targets under this owner. They must not create independent placement solvers or competing residency accounting.

## Steps

1. Fix routing-profile.sh so the full-tensor dump captures ffn_moe_topk without violating meta-backend callback invariants.
2. Profile code, prose/chat, long-context retrieval, production prompts and MTP generation; decode and prefill separately.
3. Emit hits[l][e], weight_sum[l][e], tokens, ubatch_present[l][e], assigned_tokens[l][e].
4. Measure non-expert peak VRAM at production configuration and derive per-device expert budgets from measured headroom.
5. Solver: assign `(layer,expert)` by avoided critical-path service time per resident byte. Include device bandwidth, PCIe upload latency, no-P2P topology and prefill `ubatch_present`; target CPU route mass <=0.5%, preferably <=0.25%.
6. Placement JSON v1 includes model fingerprint, architecture, expert layout/quant, placement generation, n_layer/n_expert and per-expert tier/device. Reject wrong fingerprints, duplicate/missing experts and impossible devices.
7. Persistence experiment: keep physical residency warm across requests while prompt-local predictors may reset. Prefetch only when measured lead time can hide transfer.
8. Qualify upstream llama.cpp #29887 at equal VRAM: whole-layer baseline, pure LRU, static hot, hybrid static+LRU. Preserve its <=32-token gate initially.
9. Consolidate MET05/1328 into the same solver: ROCm3 auxiliary residency is a third GPU tier, not a separate whole-layer policy. First keep 1328's safe whole-layer implementation; then, only after correctness, allow the solver to emit expert-granular ROCm3 candidates for a future extension.
10. Add a **miss-cost-aware hybrid objective**. A host expert that is repeatedly uploaded by #29887 may be cheaper as persistent ROCm3 residency even when ROCm3 compute is slower than XTX. Conversely a rare expert should remain host/LRU rather than consume 6900 VRAM. Score each candidate against its measured alternative, not nominal FLOPS.
11. Instrument one canonical residency table with counters: static hit, LRU hit, aux hit, host miss, uploaded bytes, upload stall, aux staging bytes/stall, eviction, and useful prefetch. Aggregate by `(layer,expert,device)` and placement generation.
12. Keep large-batch prefill separate. #29887 intentionally bypasses cache above 32 tokens and its published data shows prompt regressions when cache VRAM displaces whole resident layers; the solver may allocate different decode and prefill budgets but must use the same physical residency/accounting owner.

## Detailed Solution & Technical Design

### Canonical tier decision

For each `(layer,expert)` compare expected service cost under mutually exclusive tiers:

```text
cost(host) = p_active * (cpu_service_ms + required_sync_ms)
cost(lru_gpu_d) = p_active * (p_hit * gpu_service_ms[d]
                    + p_miss * (h2d_ms[d] + gpu_service_ms[d]))
cost(aux6900) = p_active * (host_stage_in_ms + aux_service_ms + host_stage_out_ms)
cost(static_gpu_d) = p_active * gpu_service_ms[d]

score(tier) = baseline_cost - cost(tier)
              - lambda_bytes * resident_bytes
              - lambda_move * migration_ms
              + stickiness * already_resident
```

Use measured critical-path costs. Do not substitute theoretical bandwidth for `h2d_ms`, aux staging or compute. The no-P2P topology means every cache bank is per physical device and every upload goes host->target GPU.

### #29887 mechanism and consolidation

Upstream #29887 ports a qvac-fabric MoE cache. Scheduler callbacks resolve host `MUL_MAT_ID` weights to a persistent GPU cache, prepare only selected misses, and remap expert ids to cache slots. Different layouts have separate banks and batches >32 bypass the cache. This is the correct reuse point: if adopted, BigCherry should feed its cache capacity/residency decisions from MET rather than implement another cache allocator.

Published Qwen3.8-Flash-Next Q4_0 data is strong for decode but exposes the budget tradeoff: RTX 4090 25.0 -> 39.4/40.7 t/s (1.57-1.62x, 72-77% hits) and RTX 5090 30.8 -> 54.5/67.8 t/s (1.77-2.20x, 77-89% hits). However RTX 4090 pp512/2048/8192 falls to 0.89/0.93/0.91x with a 6.4 GiB cache and 0.70/0.78/0.76x with a 10.7 GiB cache because cache VRAM displaces whole resident expert layers. Therefore hit rate is not the objective; **avoided critical-path milliseconds per reserved GiB** is.

The submitted scheduler implementation also currently requires `sched->n_copies == 1` for cache activation. BigCherry uses graph/meta machinery heavily, so qualification must explicitly prove the cache is active in the production scheduler configuration rather than infer it from the CLI flag. Record resolved cache entries and miss uploads at runtime.

### MET05 / 1328 boundary

1328 already supplies a correctness-constrained auxiliary ROCm3 backend outside Meta/RCCL, with pinned-host staging and whole routed-expert layer placement. Do not duplicate its transport in MET01. MET01 chooses residency; MET05 owns aux execution/transport semantics. Initial comparison is whole-layer aux versus host/LRU/static-XTX at equal VRAM. Expert-granular aux placement is a later capability only if whole-layer data proves positive service-time/GiB and the scheduler can preserve routed/shared semantics without multiplying split boundaries.

A useful follow-up is a **two-level tail**:

```text
XTX/R9700 static-hot experts
    -> ROCm3 persistent warm tail
    -> per-target-GPU #29887 LRU for remaining host misses
    -> CPU fallback
```

Do not implement all levels first. Measure each transition's avoided stall. Promote the next tier only if it reduces end-to-end decode/MTP wall time after staging/synchronization.

## Files

`tools/lab/flash-next/routing-profile.sh`, `tools/lab/flash-next/expert-placement/`; existing expert loader/transport; MET05/1328 for aux execution; upstream llama.cpp #29887 for demand-cache implementation reference.

## Validation

Profiles from >=4 workloads. Solver output passes schema/fingerprint/device validation. Equal-VRAM policy matrix on gfx1100/gfx1201, then production 2xXTX+R9700 with ROCm3 auxiliary lane where applicable. Test decode/MTP <=32 independently from pp512/2048/8192.

For each policy report static GiB, LRU GiB, aux GiB, route mass by tier, cache hit rate, H2D bytes/token, aux staged bytes/token, miss-upload ms/token, aux service/staging ms/token, CPU fallback ms/token, TG/effective TG, PP and peak VRAM. Include cold start, repeated warm requests and workload shift.

Runtime gate for #29887: prove scheduler cache callbacks actually resolve/prepare entries under the production graph configuration; zero cache activity is a failed experiment, not a performance result.

## Effort & Risk

Medium policy/tooling risk; high runtime risk for asynchronous uploads and expert-granular aux execution. #29887 has strong NVIDIA decode evidence but no local AMD proof. 1328 is already the canonical aux transport/correctness owner and remains whole-layer until hardware evidence justifies finer granularity.

## Standards

One canonical placement/residency accounting surface; policy/transport separation; generation-safe async work; reuse upstream cache machinery before forking; equal-VRAM comparisons; no-P2P-safe device ownership; optimize critical-path service time/GiB rather than hit rate.

## Acceptance Criteria

- Profiles cover >=4 workloads and decode/prefill routing.
- Placement remains within measured VRAM/headroom budgets and target CPU route mass.
- Static, LRU, hybrid and aux policies are compared at equal expert-VRAM budgets.
- A promoted #29887-style cache reuses upstream scheduler/cache machinery and proves nonzero cache activity in production scheduler mode.
- MET05/1328 remains the sole owner of ROCm3 aux execution/host staging; MET01 only supplies policy/placement.
- Any expert-granular aux extension requires whole-layer 1328 evidence showing positive end-to-end service-time/GiB first.
- Large-batch prompt regressions are included; decode hit rate alone cannot promote a feature.
- Warm residency/prefetch is promoted only when migrated bytes or transfer stalls fall without material tail-latency regression.

## Notes

Key CPU sensitivity: ~2 ms per CPU-active layer; with 10 selected experts, small CPU route mass can create large layer-touch probability. Route mass, not expert count, is the governing quantity.

2026-10-02 first routing profile: Flash-Next UD-IQ4_XS, one XTX + routed experts on CPU, ub512, mixed prose+code prefill. Per layer mean 496/512 experts used; hottest 10/25/50% of experts cover 41.5/68.8/91.3% of picks. Experts needed for 50/80/90/95% picks: mean 73/174/235/287. A <=0.5% CPU route-mass budget is therefore tight at Q6 and motivates measured multi-tier residency rather than expert-count heuristics.

External references verified 2026-10-05:
- DwarfStar persistence/prefetch mechanism: https://github.com/antirez/ds4/blob/0aaea5a238fb41a35106a551e73c8409dfb751ac/ds4_gpu.h
- llama.cpp #29887: https://github.com/ggml-org/llama.cpp/pull/29887 ; updated 2026-10-04T21:51Z, still open. Submitted implementation adds scheduler-level MoE cache callbacks, per-layout cache banks, selected-expert miss uploads and <=32-token gating.
- llama.cpp release baseline visible 2026-10-05: b11396 (`2e7c58c`, released 2026-10-04 17:45 UTC). #29887 is newer/open work and must not be treated as released baseline behavior.

## Change Log

- 2026-10-02T04:44:44.458623+00:00 (created-by): Created by agent
- 2026-10-02T06:35:13.803287+00:00 (updated-by): Updated: section:notes
- 2026-10-04: Consolidated DwarfStar persistence and llama.cpp #29887 into MET01.
- 2026-10-05: Consolidated MET05 auxiliary residency with the canonical solver; added miss-cost-aware static/LRU/aux policy, scheduler-activation gate, and two-level tail experiment.
