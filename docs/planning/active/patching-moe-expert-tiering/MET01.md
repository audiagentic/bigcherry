---
id: MET01
order: 1
plan: patching-moe-expert-tiering
state: pending
created-at: '2026-10-02T04:44:44.458623+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# Per-layer routed-expert profile + canonical residency policy

## Description

MET01 is the single policy/accounting owner for routed-expert residency. It profiles routing, derives measured per-device expert budgets, and chooses among whole-layer residency, static-hot experts, adaptive hot-expert residency, upstream demand cache, ROCm3 auxiliary residency, and host fallback. It must not own a second scheduler, cache transport, auxiliary transport, or loader.

The immediate integration gate remains upstream llama.cpp #29943 + #29887. #29943 moves selective host-expert copying out of generic `ggml_backend_sched_compute_splits()` into a public scheduler copy callback. #29887 is explicitly intended to become user-code-only after that refactor. BigCherry should therefore qualify the callback boundary before carrying any private scheduler cache fork.

The 2026-10-06 audit adds one narrow requirement: static route frequency is not enough to choose a decode cache. Recent AMD llama.cpp fork evidence shows that an adaptive cache can exploit short-term routing locality, but also shows how easy it is to report false speedups when batch kernels bypass the cache/pointer-table contract. BigCherry must first measure temporal locality and prove every batch/quant path correct; only then compare static versus adaptive admission.

## Repository evidence

- Existing routing evidence on Flash-Next UD-IQ4_XS, ub512: mean 496/512 experts touched per layer in mixed prose+code prefill; hottest 10/25/50% cover 41.5/68.8/91.3% of picks; mean experts needed for 50/80/90/95% picks are 73/174/235/287. Static expert-count heuristics are therefore insufficient.
- `tools/lab/flash-next/routing-profile.sh` currently records aggregate per-layer counts from one mixed prefill run. It does not preserve token order, adjacent-token overlap, reuse distance, workload-shift convergence, or decode-vs-prefill temporal behaviour. Those are now required before adaptive-cache implementation.
- MET05/patch 1328 already owns auxiliary 6900 execution and host staging. MET01 may select that tier but must not reproduce its transport.
- MET06 owns standard-GGUF materialization only if normal host-weight execution/cache mechanisms leave a measured placement gap.
- No production BigCherry adaptive expert cache exists yet; this audit does not treat external fork results as BigCherry measurements.

## Upstream mechanism: #29943 + #29887

#29943 adds `ggml_backend_sched_copy_callback` in `ggml/include/ggml-backend.h`, stores it on `ggml_backend_sched`, and funnels split inputs through `ggml_backend_sched_copy_input()` in `ggml/src/ggml-backend.cpp`. Non-weight inputs are copied first; host-backed weight inputs are copied last so user code can inspect already-copied routing inputs. If the callback returns false, the scheduler performs the normal whole-input copy.

The current #29943 diff removes the scheduler's embedded `MUL_MAT_ID` expert-selection implementation and installs `llama_context::sched_copy_experts` from `src/llama-context.cpp`. This is the consolidation boundary: selective expert-copy/cache policy belongs above generic scheduler machinery. Do not move adaptive policy into `ggml_backend_sched_compute_splits()`.

#29887 supplies a persistent-cache candidate: per-layout GPU cache banks, selected-expert miss upload, remapped expert IDs and a small-batch cache gate. Published NVIDIA results are mechanism evidence only; AMD promotion requires gfx1100/gfx1201 measurements.

## External AMD discriminator: moe-hotcache

`ap03906101/moe-hotcache` is useful because it exercises a closely related design on ROCm rather than because BigCherry should import the fork. Its relevant mechanism is:

1. host-resident expert weights plus a per-expert device pointer table;
2. `mmvq` dereferences either a VRAM cache slot or pinned-host/UVA address;
3. a graph-captured device counter records routed expert use;
4. a background host thread periodically drains counters, applies decay and promotes a bounded number of experts;
5. promotion copies into a free slot before pointer publication; eviction redirects to the host pointer before a one-cycle slot quarantine;
6. budget is divided per tensor rather than first-come-first-served.

Important measured evidence is narrower than the fork headline. On its Qwen3.8-Flash-Next 131B Q3_K_XL / gfx1200 system, the quality-verified partial-pin decode result is 11.4 -> 13.0-13.4 tok/s (~14-18%). Earlier larger dynamic-cache numbers were explicitly retracted after output corruption was discovered. Prefill claims were also retracted; the correct prefill path was slower than the tuned baseline. Therefore BigCherry must use this fork primarily as a correctness/architecture lesson, not as a performance promise.

The fork found two constraints directly relevant to BigCherry:

- graph replay means host-side per-token hooks run at capture, not every replay; per-token counting must be a captured device operation or another graph-safe primitive, while adaptive control must be asynchronous/out-of-band;
- its pointer-table path covered `mmvq` but not all MMQ/batch paths, producing plausible-speed corrupt output. Cache qualification must be explicit by batch regime and quant type, including prefill.

It also found a chipset-limited second GPU (~3.5 GB/s effective) was worse as a dynamic cache tier than host RAM. That reinforces MET05's existing rule: the PCH 6900 may be useful for persistent whole-layer work, but must not be assumed useful for miss-driven expert traffic.

## Implementation plan

1. **Extend profiling before cache code.** Extend `tools/lab/flash-next/routing-profile.sh` or its parser so the canonical trace preserves `(request, phase, token/ubatch, layer, selected expert ids, route weights)` rather than only aggregate counts. Emit aggregate `hits[l][e]` plus adjacent-token overlap, reuse-distance histogram, working-set size over 8/32/128/512-token windows, and workload-shift convergence. Capture decode and prefill separately across code, chat/prose, retrieval/long-context and MTP.
2. **Cheap offline discriminator.** Replay those traces through static-hot, LRU, decayed-frequency and bounded-promotion policies without touching llama.cpp. For equal slots/GiB report hit rate, bytes avoided, promotions/token and convergence after a workload switch. If adaptive policy cannot beat static-hot by >=5 percentage points of miss bytes or a measured-cost equivalent on at least two decode workloads, reject adaptive implementation and retain static/upstream LRU qualification only.
3. **Adopt the callback seam, not a private scheduler cache.** On the current pin, determine whether #29943 is present. If absent, qualify a minimal backport containing only the public copy-callback seam and moved selective-copy logic. Do not add MET-specific policy to `ggml/src/ggml-backend.cpp`.
4. **Mock callback correctness before cache policy.** Install an observation-only callback returning false for every host weight while counting callback invocations, backend, tensor bytes and graph first-op. Require byte/token/logit identity with callback disabled. Then enable upstream selective copy and require identical greedy output plus nonzero selected-copy counters.
5. **Qualify persistent cache at equal VRAM.** Compare whole-layer baseline, pure LRU, static-hot and hybrid static+LRU first. Preserve upstream small-batch gating initially. Prove cache activity under production scheduler copies; zero activity is a failed experiment.
6. **Adaptive cache only after Step 2 passes.** Reuse the same cache bank/slot map as the persistent cache. Add no second cache allocator. Device-side route counters must be graph-captured/replay-safe. A host controller may update policy between replays/requests, but slot publication must obey copy-complete-before-publish and redirect-before-reuse lifetime rules. Use bounded promotions to cap H2D bandwidth.
7. **Batch/quant dispatch contract.** Enumerate every execution path that can consume cached/host expert weights (`mmvq`, MMQ/grouped MMQ, prefill paths and MTP verify shapes). A pointer-table or remap representation may be enabled only where the consuming kernel explicitly supports it. Unsupported shapes must use upstream selective copy/CPU fallback; never let `supports_buft` or placement metadata silently route an incompatible kernel to raw host/cache pointers.
8. **Canonical objective.** Rank each `(layer,expert,tier)` by avoided critical-path milliseconds per resident byte, using measured H2D, compute, sync and auxiliary staging costs. Adaptive hit rate is an input, not the objective.
9. **Aux tier.** Feed MET05/1328 candidates from the same placement JSON. Keep 1328 whole-layer until hardware evidence proves expert-granular auxiliary placement worthwhile.
10. **Large-batch separation.** Keep pp512/2048/8192 separate from decode/MTP. Prefill often touches nearly all experts; a decode cache that displaces whole resident layers or causes host-page pressure is not globally promoted. Workload-specific policy is allowed only with explicit budgets and fail-closed dispatch.
11. **Only then consider MET06 slicing.** Runtime standard-GGUF slicing remains blocked unless callback/cache/static/aux controls leave >=5% TG/PP opportunity attributable to coarse placement or >=1 GiB avoidable resident expert memory at equal throughput.

## Offline policy replay pseudocode

```text
for event in trace_in_token_order:
    for (layer, expert) in event.routes:
        record_hit_or_miss(policy[layer], expert)
        policy[layer].observe(expert)
    if policy.refresh_due(event):
        candidates = policy.rank_promotions()
        promote_at_most(K, candidates)   # account bytes, do not assume free transfer

report per workload/window:
    hit_bytes, miss_bytes, promotions, churn, convergence_tokens
```

The offline replay is deliberately policy-only. It must not model a hit as free: convert hit/miss counts to predicted critical-path time only using measured local H2D/host-execution costs, and label that result as a prediction until hardware ABBA exists.

## Ownership and files

- `tools/lab/flash-next/routing-profile.sh`, `tools/lab/flash-next/expert-placement/`: canonical trace/profiling/solver and offline policy replay.
- `ggml/include/ggml-backend.h`, `ggml/src/ggml-backend.cpp`: upstream #29943 callback seam only; no MET policy.
- `src/llama-context.cpp` / context-owned helper: selective-copy/cache user-code owner from #29943/#29887.
- Backend MMVQ/MMQ files: only if an already-selected cache representation needs explicit consumer support; no policy ownership in kernels.
- MET05 / patch 1328: auxiliary ROCm3 execution and staging.
- MET06: standard-GGUF materialization fallback only after the measured gate.

No second dispatch table, residency map, cache allocator, prefetch scheduler or ROCm3 transport is permitted. Static/LRU/adaptive policies must share one cache-bank representation and one accounting schema.

## Validation matrix

Hardware: gfx1100 XTX, gfx1201 R9700, then production 2xXTX+R9700; 6900 only through MET05 controls. No-P2P means every primary cache bank/upload is target-device-local.

Correctness before performance:
- callback-disabled vs observation-only callback: greedy identity and logits/KLD contract;
- selective copy/cache: multi-request same-process, cold->warm->workload-shift, MTP on/off;
- explicit batch boundaries around every dispatch transition (at minimum 1/2/4/8/9/16/32 plus production ubatches where applicable);
- representative quant/layout coverage for the production Flash-Next GGUF, not one tensor type only;
- pp512/2048/8192 bypass/cache lanes and long-context scheduler-copy reuse;
- cache counters prove selected experts, uploaded bytes, promotions and slot generations are nonzero and physically plausible;
- deliberate negative test: disable/omit cache support for one batch path and prove dispatch falls back rather than reading the wrong representation.

Performance: ABBA >=5 repetitions for TG128/512 and PP512/2048/8192; report median/dispersion, static/LRU/adaptive/aux GiB, route mass by tier, cache hit, H2D bytes/token, promotion bytes/token, miss-upload ms/token, auxiliary service/staging, CPU fallback, peak VRAM and locked host memory.

Promotion: >=5% end-to-end TG/effective-TG or PP improvement at equal expert-VRAM budget with <=2% regression in the unaffected regime. Adaptive policy additionally must beat the best simpler static/LRU policy by >=3% end-to-end TG on at least two decode workloads after promotion traffic is included. Otherwise retain the simpler policy. A workload-specific decode policy may be retained if prompt regression is explicitly isolated and feature-gated.

Reject any result whose implied transfer/work exceeds measured physical limits, whose output/correctness gate fails, or whose speedup disappears when the same number of expert operations is accounted for.

## Acceptance Criteria

- One canonical placement/accounting schema covers static, LRU, adaptive, aux and host tiers.
- Routing evidence includes temporal locality and workload shifts, not only aggregate expert counts.
- Offline trace replay eliminates noncompetitive adaptive policies before runtime implementation.
- #29943 callback seam is either present upstream or qualified as a minimal temporary backport; MET logic does not live in generic scheduler code.
- Observation-only callback proves semantic transparency before selective-copy/cache testing.
- Persistent cache proves nonzero activity on gfx1100/gfx1201 and passes batch/quant/multi-request integrity gates.
- Adaptive refresh is graph-safe and reuses the same cache bank; no host hook is assumed to execute per graph replay.
- Static/LRU/adaptive/aux are compared at equal expert-VRAM budget; decode and large-batch prompt effects are both reported.
- MET05 remains sole auxiliary transport owner; MET06 remains blocked until its explicit placement-gap gate passes.
- Unsupported or inactive paths fail closed rather than silently becoming baseline measurements.

## Notes

2026-10-05 audit: #29943 materially improves the integration boundary for MET. Its diff removes expert-ID parsing/copy grouping from generic scheduler compute and exposes a host-weight copy callback, while `llama_context` becomes the user-code owner. #29887 states that after #29943 it should be entirely user-code. This reduces BigCherry's reason to carry scheduler-core MoE cache patches and makes the first local mock cheap: an observation-only callback can validate ordering/lifetime on HIP before cache code is introduced.

2026-10-06 audit: AMD `moe-hotcache` adds useful negative evidence. Its quality-verified Qwen3.8 result supports testing adaptive decode residency on ROCm, but its larger published gains and prefill gains were retracted after batch-path output corruption. BigCherry therefore adds temporal trace replay and explicit batch/quant dispatch correctness as prerequisites rather than porting the fork. The fork's graph-replay lesson also constrains implementation: device counting must live in captured work; adaptive host policy must not rely on a per-token host callback.

External references:
- llama.cpp #29943 `ggml: refactor selective expert copying to user code` (open/current diff inspected 2026-10-06).
- llama.cpp #29887 `add a GPU cache for MoE experts kept in host memory`.
- `ap03906101/moe-hotcache` main at `beda4496714f03e74bded372402cba5224e29baf` inspected 2026-10-06; ROCm adaptive-cache architecture and explicit retractions/correctness lessons.
- `Maxritz/Strata-rocm` main at `2ed00de617659e3e238bdc0ae7523e1e99ef4e38` inspected 2026-10-06; its ROCm port reports PCIe-saturated miss handling and rejected double-buffer overlap when link contention made it slower, reinforcing measured-cost rather than overlap-by-default policy.

## Change Log

- 2026-10-02T04:44:44.458623+00:00: created.
- 2026-10-04: consolidated DwarfStar persistence and #29887 into MET01.
- 2026-10-05: consolidated MET05 auxiliary residency with canonical solver.
- 2026-10-05: made #29943 user-code copy callback the required integration seam; added observation-only mock, HIP qualification matrix and explicit no-duplicate-scheduler boundary.
- 2026-10-06: added graph-safe adaptive-cache discriminator, temporal routing trace requirements, batch/quant correctness gates, and AMD fork negative evidence.
