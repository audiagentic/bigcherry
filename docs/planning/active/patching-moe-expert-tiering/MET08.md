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

**2026-10-17 reconciliation of the published BCOP241 review.** The 128/128/256 full-context fit with patch 1341 was completed on 2026-10-07 (recorded below); do not rerun that historical step or create a second reordered GGUF before the following integrity gates. No damaged GGUF was observed.

1. **Fail closed before modifying any source copy:** in the existing `tools/lab/flash-next/expert-place.py`, validate complete/unique per-layer tensor names, shard inventory, exact separate `ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps` and router weights required by patch 1283; reject fused-only `ffn_gate_up_exps`. Validate STRP version, expert-ID permutation, finite positive traffic weights, nonnegative integer capacities summing to total experts and block-byte divisibility. Current `len(names)<4 and gate_up not in names` check occurs **after** copies/writes and admits unsupported fused-only layers and duplicates.
2. **Transactional output:** make derived shards in a disposable sibling staging directory; verify all layers, inverse mapping, unchanged non-target bytes, hashes and a provenance manifest (source/model/profile hashes, quantizer/pin, capacity split, `--traffic`). Publish atomically only after checks; failure must leave existing output untouched. Cover missing/duplicate/merged tensors, truncated STRP, invalid weights, incomplete shards and interrupted staging with host negative fixtures.
3. **Admission before another large copy or GPU lane:** use existing 1338 routing receipts and `tools/lab/strata/routing-balance.py` to compare held-out 1:1:1 versus proposed 2:2:1 *executed work and measured per-rank service*, not just rank assignment counts. If exposed critical-path E2E improvement cannot plausibly exceed 3%, close the traffic experiment without generating another model artifact.
4. **Conditional hardware:** first resolve MET09's first-divergence/physical-ID tie issue. Then use same-original-file controls, matched context/ubatch/verify work, multi-prompt same-process replay, full logits/KLD/top-k/greedy and acceptance accounting, physical transfers/VRAM and paired ABBA. Promote opt-in only with reproducible CI95-low >=3% end-to-end improvement and <=1% other-regime regression; otherwise reject.
5. **Ownership:** MET08 is sole offline-writer owner; MET11/B2 mapping must wait for an integrity-verified oracle, MET09 owns divergence and MET04/1283 owns expert-axis execution. Do not duplicate the writer, placement map, scheduler, allocator or hardware queue.

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
