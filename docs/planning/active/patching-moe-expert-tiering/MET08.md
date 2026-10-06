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
