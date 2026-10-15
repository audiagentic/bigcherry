---
id: MET08
order: 8
plan: patching-moe-expert-tiering
state: pending
created-at: '2026-10-06T14:53:05.705421+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P1
work: M
---

# Usage-placed expert layout: dense + attention on the XTXs, hot experts across all three cards, cold on the R9700

## 2026-10-16 BCOP241 implementation audit — offline placement integrity before 2:2:1

**Disposition:** PENDING, fail-closed derived-GGUF artifact gate before any new model copy or GPU lane. The 128/128/256 + 1341 full-context fit was completed 2026-10-07; original Step 1 below is historical. MET08 owns offline permutation/placement qualification; MET11 owns eventual load-time B2; MET09 owns first-divergence diagnosis; MET04/1283 owns expert-axis execution. MET07's native MoE cache is separate.

### Exact source and correctness finding

- `tools/lab/flash-next/expert-place.py::read_profile` reads ranked STRP expert IDs; `place` uses credit-based weighted round-robin on **rank positions**, not measured expert counts, co-occurrence or GPU service times. For 512 experts, caps 128/128/256, traffic 2:2:1, the first 320 ranked experts are assigned 128/128/64. This is a host-only calculation, not an observed 2:2:1 service ratio or performance gain. The older 1:1:1 trace (MET04) reports 3.31/3.31/3.38 experts/token, busiest rank 4.85 in-sample; it does not predict the new policy's critical path.
- `expert-place.py::main` copies each GGUF shard, then modifies its expert/router bytes with `r+b`, and checks completeness only **after** all shards. Its current condition `len(names) < 4 and 'ffn_gate_up_exps.weight' not in names` accepts a layer containing only `ffn_gate_up_exps.weight`, or four duplicate names. However `1283_qwen4exp_expert_parallel/patch.py` matches only separate `ffn_(gate|up|down)_exps.weight` and explicitly excludes merged gate_up. A partially transformed or unsupported derived GGUF can therefore pass this check. **Source-confirmed fail-open risk; no corrupted model was observed.** The tool's lossless-byte-permutation claim also does not imply identical routing: B1 permutes router rows and fused topk_moe's physical-index tie policy can select a different original expert (MET09).
- Patch 1283 remains a validated, default-off expert-axis enabler (historical b11402 README receipts), not an unimplemented phase-B proposal. Its SUMMARY's stale pending evidence is corrected here. Do not infer execution from placement-map existence; require actual range-MUL_MAT_ID and final single-AllReduce evidence.

### Bounded implementation-ready gates

1. **Before writing:** extend only the existing `expert-place.py` to check STRP length/version, layer and expert bounds, unique IDs, finite positive traffic, nonnegative integer caps summing to n_expert, complete source shard set, unique per-layer tensor names, block divisibility and the exact supported family. For current 1283 Qwen4Exp require gate/up/down/router **exactly once**; reject gate_up until 1283 supports it. Do not create another writer, cache or placement table.
2. **Transactional derived file:** copy to a disposable sibling directory; verify all permuted blocks, inverse map, unchanged non-target bytes and shard counts; record source/destination SHA256, profile hash, pin, caps/traffic and exact per-layer tensor names in `expert-placement.json`. Atomically publish only after full verification. On failure remove staging, leave prior artifact untouched. Negative tests: missing router/down, fused-only, duplicates, truncated/invalid STRP, out-of-range ID, NaN/Inf traffic, negative cap, missing shard, interrupted staging.
3. **Cheapest performance discriminator:** use existing 1338 STRP and `tools/lab/strata/routing-balance.py` to compare 1:1:1 and 2:2:1 on held-out per-token routes at identical caps. Report selected experts/token per physical rank, P50/P95 busiest rank, expected/observed work and a bound weighted by *measured* gfx1100/gfx1201 per-expert service; do not mistake rank counts for latency. No held-out trace or less than 3% exposed E2E theoretical headroom => `NO_ADMISSIBLE_GAIN`; do not create another 88 GB model.
4. **Only if admitted:** MET09 first-divergence gate, then same-pin same-original-GGUF A/B at 8K/24K/98K/202K and ctx245760, fixed KV/ubatch/ranks. Compare original row split, validated B1 derived GGUF, 1283 flag-off and 2:2:1; MTP-off fixed-work verify ms is primary, MTP-on acceptance secondary. Include multiple prompts, two same-process requests, long-context graph replay, full logits/KLD/greedy, selected expert work, per-rank kernel/AllReduce/PCIe, VRAM, four independent ABBA sessions. Reordered-file row-split text differences and changed MTP acceptance are confounds, not performance proof. Promote only if CI95-low >=3% E2E, <=1% other-regime regression and no correctness/work discrepancy; otherwise reject.
5. MET11 B2 must keep original router/top-k slot weights, translate only expert-dispatch old IDs through a graph-owned mirrored map, and preserve request/replay lifetime. Do not equate B1 and B2 bit identity at score ties or allocate a patch number before these gates.

**Measured historical evidence (not rerun):** 2026-10-07 128/128/256 with 1341 fits 245760; 98K decode 60.7/60.6 versus 59.2/58.9 t/s with different draft acceptance. The 132/132/248 variant lost at 98K despite improving at 8K, so acceptance-dependent decode rates cannot attribute layout speed. No new GPU result.

**External mechanisms:** [llama.cpp #29887](https://github.com/ggml-org/llama.cpp/pull/29887) merged 2026-10-07 and provides a small-batch native MoE cache (MET07-owned), not static permutation. [vLLM ExpertMapManager](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/expert_map_manager.py) separates global/local/physical IDs and kernel support; [SGLang EPLB](https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/expert_parallelism.mdx) balances from actual activations. They are reference mechanisms, not transferable AMD gains. Host-only 7/7 source-shaped tests and 12/12 repository assertions passed; no GGUF writer execution, HIP build or GPU benchmark.

## Description

Owner's layout (2026-10-06): tensor-split the dense weights and attention (all KV) on the two XTXs, deal each layer's high-usage experts over all three cards leaving room on the XTXs, put the cold remainder on the R9700. Built from 1283 (expert split + BIGCHERRY_MOE_EP_TS), tools/lab/flash-next/expert-place.py (lossless per-layer expert reorder from a usage profile) and -ts 1,1,0. Measurements so far are recorded in MET04: prefill equal to +1% against the row split; decode a few percent higher but following draft acceptance, which changes with the partition; 128/128/256 did not fit at ctx 245760 before 1341 (R9700 full), 132/132/248 and 136/136/240 fit.

## Steps

1. Re-run 128/128/256 at ctx 245760 with 1341 on (1440 MiB returned on the R9700). 2. A decode measure that does not depend on acceptance: several prompts (code, prose, tool JSON) and ms per verify step, ABBA against production on the ORIGINAL file. 3. Hot traffic weighted to the XTXs (2:2:1) - needs a second reordered copy (disk: 41 GB free, remove the 1:1:1 copy first). 4. Explain why the row split gives a different text on the reordered file (MET09). 5. Per-device selected-expert counters in 1283 so balance is measured inside the split. 6. Decide: adopt as a deployment layout (reordered model file + profile flags), keep as an option, or reject.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files

patches/1283_qwen4exp_expert_parallel, tools/lab/flash-next/expert-place.py, queue-moe-ep.sh, gpu-usage-sampler.sh, tools/lab/strata/routing-balance.py

## Validation

Adoption needs: fits at the production context, prefill not below production at 8K / 98K / 202K, decode not below production on a multi-prompt set, fidelity inside the envelope at each depth, and a reproducible recipe for the reordered file (profile + caps + traffic recorded in expert-placement.json).

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Depends on MSM02 / MSM03 for memory headroom on the R9700. The reordered file is a derived artifact (gguf/placed-128-128-256), not a new quantisation.

2026-10-07 (build b-metamem-b11402h2, 1341 on). STEP 1 DONE: the owner's layout at the intended shares 128/128/256 now loads and runs at ctx 245760 with BIGCHERRY_META_SUBSET_MIRROR=1 (it ran out of memory on the R9700 before 1341). VRAM peak at 98K depth: 23.5 / 23.5 / 31.4 GB (production control on the same file: 23.8 / 24.1 / 30.8). 8K: prefill 955.0/1068.4 vs 977.4/1056.7 (first run of each arm is a cold outlier; equal), decode 84.2/84.8 vs 80.9/81.7, acceptance 351/480 vs 344/498, 345/495. 98K: prefill 975.6/974.6 vs 956.5/988.2, decode 60.7/60.6 vs 59.2/58.9, acceptance 355/466, 354/469 vs 350/482. Fidelity 22/24, TV 0.080. Same picture as at the other contexts: prefill equal, decode a few percent up with slightly higher acceptance. EQUAL DENSE/EXPERT SHARES ON THE PRODUCTION ROW SPLIT (-ts 1,1,1 against 0.31,0.27,0.42; not usage-placed, original file): ctx 49152: 8K prefill 1119.4/1115.7 vs 1074.1/1075.5 (+4%), decode 89.6/90.8 vs 85.2/86.3 (+4..5%); 24K prefill 1119.5/1119.6 vs 1077.6/1077.5 (+4%), decode 78.9/79.2 vs 76.0/76.4 (+3.5%); acceptance equal; probes vs production top-1 21/24, TV 0.124 (a different split, as expected). At ctx 245760 equal shares do NOT load (both depths). So production's skewed shares cost about 4% in both prefill and decode and exist only for memory at the full context; a split between the two that fits at 245K (1341 freed 1440 MiB on the R9700, but the XTXs hold the KV and are the limit) is worth a short fit sweep, and the owner's layout is one way to get balanced expert load without putting more bytes on the XTXs. Swift files and Strata packs were removed for disk space (2026-10-07); the reordered UD copy gguf/placed-128-128-256 is kept.

## Change Log

- 2026-10-06T14:53:05.705421+00:00 (created-by): Created by agent
- 2026-10-06T16:35:25.414709+00:00 (updated-by): Updated: section:notes
