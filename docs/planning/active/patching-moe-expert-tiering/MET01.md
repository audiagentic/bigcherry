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

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files

tools/lab/flash-next/routing-profile.sh, tools/lab/flash-next/expert-placement/ (profile aggregation, solver, validator)

## Validation

Profiles from >=4 workloads; solver output passes validator; predicted CPU-active layers/token reported; budgets derived from measured free VRAM, not nominal capacity.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Key number: CPU cost ~2 ms per CPU-active layer; P(layer touches CPU) = 1-(1-q)^10 for CPU route mass q. 1% CPU mass ~ +9 ms/token vs 23.3 ms/token MTP baseline. Route mass, not expert count, decides CPU cost. Prefill activates far more cold experts (512*10 selections/ubatch).

## Change Log

- 2026-10-02T04:44:44.458623+00:00 (created-by): Created by agent
