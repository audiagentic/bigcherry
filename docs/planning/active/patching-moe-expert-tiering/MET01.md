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

Prerequisite for every tiering patch. Collect per-layer routed-expert statistics for qwen4exp (Flash-Next) from ffn_moe_topk and turn them into a validated placement JSON under a measured per-device VRAM budget. Design agreed with GPT 2026-10-02 (session ses_f4675ceea21f47bf, req_6aee98215e88444d).

## Steps

1. Fix routing-profile.sh (eval-callback rejects --tensor-filter) so 1279's full-tensor dump captures ffn_moe_topk.
2. Profile workloads: code, prose/chat, long-context retrieval, production prompts, MTP generation; decode and prefill separately.
3. Emit hits[l][e], weight_sum[l][e], tokens, ubatch_present[l][e], assigned_tokens[l][e].
4. Measure non-routed-expert peak VRAM at production config (192K q8_0 KV, MTP depth 3, -sm tensor -ts 3,3,2, PLE CPU, all routed experts CPU) to get per-device tier budgets (headroom: XTX 2 GiB, R9700 2-3 GiB, 6900 1-2 GiB).
5. Solver: assign (layer, expert) by marginal benefit/byte, balance expected active bytes by bandwidth (960/960/640/512 discounted for 6900 transfer latency), simulate per-layer max(service times)+reduction; target CPU route mass <= 0.5% (prefer <= 0.25%); include ubatch_present for prefill.
6. Placement JSON v1 (architecture, model_fingerprint, n_layer, n_expert, tiers, per-layer tier[512]); validator rejecting wrong fingerprint/n_expert, missing/duplicate experts.
7. Add a persistence/prefetch policy experiment inspired by DwarfStar: keep the resident expert set warm across requests/session resets; maintain short/long-horizon hit EMAs; optionally prefetch only when the route decision/look-ahead provides enough lead time to hide transfer. Keep transport separate from policy and do not create a second expert store.

## Detailed Solution & Technical Design

DwarfStar's useful mechanism is policy separation: its resident expert cache remains warm across sessions while prompt-local eviction heuristics reset independently, and its implementation has optional expert look-ahead/prefetch hooks. Import that shape, not its SSD transport. BigCherry's discrete no-P2P topology should score persistence by expected route mass avoided per byte and migration cost.

Conceptual score extension:

```text
score[layer,expert,device] =
    ema_hits * avoided_service_ms
  - lambda_bytes * resident_bytes
  - lambda_move * expected_migration_ms
  + stickiness * already_resident
```

The solver remains the single placement owner. Persistence changes initial conditions/eviction cost; it must not fork a second placement algorithm. Prefetch is a separate optional action emitted by the same policy and consumed by the existing transport/loader.

## Code Samples & Guidance

Keep workload-history state compact: per-layer/expert counters/EMA plus current placement generation. Session reset may clear prompt-local predictors without discarding physical residency. Any prefetch request must carry placement generation so stale asynchronous work can be dropped safely.

## Files

tools/lab/flash-next/routing-profile.sh, tools/lab/flash-next/expert-placement/ (profile aggregation, solver, validator); existing expert loader/transport only if a later implementation experiment is promoted.

## Validation

Profiles from >=4 workloads; solver output passes validator; predicted CPU-active layers/token reported; budgets derived from measured free VRAM, not nominal capacity. Persistence experiment must compare cold-start placement vs warm cross-session placement on repeated and shifted workloads. Report expert hit mass, bytes migrated/request, transfer stall ms/token, and effective TG. Prefetch only passes if useful-hit rate and hidden-transfer time improve without increasing worst-case latency materially.

## Effort & Risk

Medium. Policy/tooling is low risk; asynchronous prefetch on discrete GPUs is higher risk because PCIe/host latency may not be hideable after routing is known. Do not implement transport until trace timing proves useful look-ahead.

## Standards

One canonical expert placement solver/store; policy/transport separation; generation-safe asynchronous work; reuse-before-fork.

## Acceptance Criteria

- Profiles cover >=4 workloads and decode/prefill routing.
- Placement remains within measured VRAM/headroom budgets and target CPU route mass.
- Warm-residency experiment proves whether cross-session stickiness reduces migrated bytes/stalls.
- Prefetch is promoted only when route lead time can hide a material part of transfer latency.
- No second expert cache/store or duplicate placement solver is introduced.

## Notes

Key number: CPU cost ~2 ms per CPU-active layer; P(layer touches CPU) = 1-(1-q)^10 for CPU route mass q. 1% CPU mass ~ +9 ms/token vs 23.3 ms/token MTP baseline. Route mass, not expert count, decides CPU cost. Prefill activates far more cold experts (512*10 selections/ubatch).

2026-10-02 first routing profile (flashnext-routing-4; Flash-Next UD-IQ4_XS, one XTX + routed experts on CPU, -sm none, because llama-debug's per-node callback asserts in the meta backend under -sm tensor; --tensor-filter exists only in llama-debug; ub 512). Single mixed prose+code prompt, 176,330 selections/layer (~17.6K tokens), prefill-only. Per layer: experts used mean 496/512 (min 293). Hottest 10/25/50% of experts cover 41.5/68.8/91.3% of picks (means). Coldest experts holding <= 0.25/0.5/1/2/5% of picks: mean 84/106/134/169/225 of 512 (min 39/54/72/95/130, max ~440 - skew varies a lot by layer). Experts needed for 50/80/90/95% of picks: mean 73/174/235/287. Implication: a <= 0.5% CPU route-mass budget puts only ~21% of routed experts (~20 GiB at Q6) in RAM; the other ~75 GiB of Q6 routed experts must sit on the four GPUs alongside ~10 GiB non-expert weights, 192K KV and the MTP draft - roughly the whole 96 GB of combined VRAM. Q6 at <= 0.5% CPU mass is at the edge; Q5_K or a 1-2% CPU budget is more realistic. Caveats: one prompt domain, prefill only (decode routing may be more skewed), IQ4_XS routing may differ slightly from Q6. Next: profile chat/long-context/MTP-generation workloads and decode-only routing.

External mechanism reference (verified 2026-10-04): DwarfStar keeps its resident SSD expert cache warm across sessions while resetting prompt-local eviction state and exposes expert look-ahead/prefetch hooks. BigCherry adopts only the residency/policy separation until discrete-GPU timing proves a transport worth implementing: https://github.com/antirez/ds4/blob/0aaea5a238fb41a35106a551e73c8409dfb751ac/ds4_gpu.h

## Change Log

- 2026-10-02T04:44:44.458623+00:00 (created-by): Created by agent
- 2026-10-02T06:35:13.803287+00:00 (updated-by): Updated: section:notes
- 2026-10-04T00:30:00+00:00 (agent): Consolidated DwarfStar warm-residency/prefetch policy into MET01 rather than creating another expert-cache plan.
