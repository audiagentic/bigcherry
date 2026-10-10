---
id: PNRO06
order: 0
plan: patching-nasone-rdna-optimizations
state: completed
created-at: '2026-09-09T10:52:38.138969+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: P0
---

# Hybrid ROCm TOP_K kernels

## Description

Disposition (2026-10-08): **closed without promotion**. Patch 1256 is a preserved, historically **untested** experiment, not a production improvement. The 2026-09-27 measured backend-sampling MTP series activated but failed the confidence gate; ordinary MoE routing did not execute generic TOP_K. More importantly, validated 1294 supplies deterministic QSA tie/ordered-output semantics and explicitly conflicts with 1256. Do not restart a third series, port another TOP_K kernel, or disable 1294 for speed.

## Steps

1. Terminal decision: no additional 1256 experiment or production recipe change on current evidence. Preserve its patch metadata/evidence for reproducibility; **completed plan** is not a claim that the patch passed validation.
2. Reopen only on a new measured production `GGML_OP_TOP_K` signature outside fused MoE routing, with a >=5% E2E wall-time contribution and a compatible deterministic-tie/ordered-output contract. Do not borrow QSA experiments from active QFP17 or 1294 owners.
3. Any successor must be a **single** opt-in, scoped selector comparison against pinned native HIP radix/bitonic and 1294, not a new general TOP_K scheduler or another wave-width table.

## Detailed Solution & Technical Design

### Implementation and dispatch audit (2026-10-08)

- Pinned `ggml-org/llama.cpp@b11474:ggml/src/ggml-cuda/top-k.cu` is byte-identical to inspected upstream master (blob `3ffbba839d6cf94c368bd291cf2d850f529de5f1`). `ggml_cuda_op_top_k()` lines 217-279 chooses native HIP `top_k_radix_cuda()` for `ncols > 1024`, otherwise bitonic argsort and D2D copy. The native radix path allocates per-row state plus `nrows * blocks_per_row * 256` histogram integers, performs radix passes and gathers; the alternative's claimed gain must include scratch, launches, barriers and copies, not only selection arithmetic.
- `1256_nro07_topk_hybrid/patch.py` ports nasone `7f3e1e4d` + `10fdba9a`: k=1 selects `topk_small`; for k>1, HIP >=7.15 can select small-row kernels, while ncols>1024 selects `topk_parallel_radix`; otherwise bitonic remains. The recorded Brutus ROCm 7.2.4/7.14 toolchains cannot activate the >=7.15 small-row route. `BIGCHERRY_PATCH_TRACE` markers prove routing, **not** numerical parity or E2E gain.
- `ggml/src/ggml-cuda/topk-moe.cu::ggml_cuda_should_use_topk_moe()` and `topk_moe_cuda()` fuse MoE gate scoring, top-k and output weights/IDs. The 2026-09-27 production MoE lane used that path and never executed the generic TOP_K being modified. Generic TOP_K is relevant to backend sampling and QSA only when the graph actually selects it.
- `1294_topk_deterministic_ties` is validated and its `patch.toml` conflicts with 1256. It replaces atomic-arrival selection at equal-score cutoffs with lowest-column tie handling and adds ordered output (2026-10-07 independent implementation). QSA often has exact zero-score ties at long context. An unqualified 1256 replacement would sacrifice this correctness property; the historical 1256 experiment is not a safe 1294 replacement.
- `1257_nro08_topk_wave32` requires 1256 and modifies the same kernels, not an independent MoE routing acceleration. PNRO07 is its subordinate disposition owner. `RNX02`/1294 own QSA tie determinism, and QFP17's active QSA-mask work is protected; do not modify their implementation, plans or queued lanes.

### First-party hardware evidence (historical, not new measurements)

| Experiment | gfx1100 point | gfx1100 CI95-low | gfx1201 point | gfx1201 CI95-low | Result |
|---|---:|---:|---:|---:|---|
| 1256 hybrid vs native, backend-sampling MTP | +1.18% | -0.67% | +1.45% | -1.79% | Not established |
| 1257 wave32 incremental vs 1256 | +0.87% | -1.17% | +0.51% | -3.09% | Not established |

Series-1 MoE did not activate generic TOP_K. Series-2 activated in 8 sessions, but the MTP lane was noisy at 10 rounds. These confidence intervals **do not** establish a positive gain, and neither experiment has production promotion evidence.

### Other engines and upstream

- Current upstream retains native HIP radix/bitonic (same source blob). No missing general TOP_K backend mechanism is established.
- vLLM ROCm AITER `rocm_aiter_grouped_topk` and SGLang's consolidated fused MoE gate/top-k are **model/consumer-specific** routing mechanisms; their advantage cannot be credited to a general `GGML_OP_TOP_K` replacement. See https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/experts/rocm_aiter_moe.py and https://github.com/sgl-project/sglang/issues/26771.
- A newer SGLang fused route+quant handoff (https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/route_quant_handoff.py) illustrates avoiding extra launches **only when downstream consumers accept packed output**. BigCherry's `topk_moe` already owns fused routing; do not introduce another generic top-k cache/selector/packing contract from this evidence.

### Reopen-only implementation gate

Require a first-party `GGML_OP_TOP_K` callsite trace with exact `ncols,nrows,k,backend,arch,toolchain,graph ID,caller`, launches, scratch bytes, and E2E contribution >=5%. Before any GPU campaign run a host/reference fixture for ncols={31,32,33,63,64,65,1024,1025,32768}, k={1,2,4,8}, rows={1,2,4,8}, ties across cutoff, +/-0, infinities, NaNs and duplicate values; preserve deterministic index ordering or explicitly reject. Compare with 1294, native HIP and the 1256 fallback; verify multi-request same-process, multi-ubatch, graph capture/replay, tensor split/rank locality, logits/greedy and MTP acceptance. Keep gfx1030/native and non-HIP unchanged. No P2P assumption: TOP_K scratch/dispatch is per owning GPU. Promotion needs >=4 independent sessions, >=10 interleaved ABBA rounds, CI95-low >=3% E2E improvement and <=1% control regression; otherwise reject the successor. This gate is **not queued**.

## Code Samples & Guidance



## Files

patches/1256_nro07_topk_hybrid/{patch.toml,patch.py,SUMMARY.md,README.md,TESTING.md}; top-k.cu; shared NRO static tests; CPU/reference TOP_K fixture runner; real signature campaign artifacts.

## Validation

Audit-only checks performed 2026-10-08: pinned/master upstream TOP_K source blob comparison (identical), patch metadata/dependency/conflict readback, first-party 2026-09-27 evidence review, and a 48-case host route-classification fixture plus a tie-arrival-order illustration. The fixture does not execute HIP kernels. No new build, hardware lane or numerical backend test ran. Terminal disposition: **no promotion**; retain historical `untested` patch state and do not misrepresent the negative-confidence campaign as a correctness failure.

## Effort & Risk



## Standards

Exact routing correctness before performance; HIP-only containment; native fallback; no conflation with downstream routing fusion.

## Acceptance Criteria

Completed as a negative optimisation decision: native HIP radix/bitonic and validated 1294 remain authoritative; 1256 is not promoted; no duplicate generic TOP_K path is introduced. Reopen only under the bounded attribution/correctness/performance gate above.

## Notes

Supersedes: NRO07
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-nasone-rdna-optimizations-nro07

2026-09-24 relevance at b11126: IMPLEMENTED-AS-PATCH. Item title says TOP_K hybrid; its own Files section and patches/ dir map it to patches/1256_nro07_topk_hybrid (state=untested) -- note the patch package is literally named nro07 (matches its "Successor key: nro07"/Supersedes NRO07 in Notes) even though the plan item id is PNRO06; this is the correct, verified mapping (grep patches/1256*/patch.toml id field confirms). No upstream HIP-native TOP_K hybrid selection kernel found in b11126 top-k.cu relevant to this scope. Disposition: validate/qualify existing patch; no GPT design needed.

2026-09-24 GPT review req_215c89d0b13a4bb7 applied: verified via grep that b11126 top-k.cu already has HIP top_k_radix_cuda (ncols>1024) and bitonic fallback with top_k_float_to_ordered -- corrected the plan's stale premise that this needed a from-scratch HIP port. Rescoped to a re-diff against the current source to find genuinely missing paths, wired ahead of the existing radix/bitonic routes under an opt-in gate, with a required dispatch marker.

2026-09-25 (ef49e4e5): the scaffold is replaced by an EXACT port of nasone 7f3e1e4d + 10fdba9a. The fork's pre-change top-k.cu is byte-identical to b11126's, so no rebase design was needed: new tools/bigcherry/patch/port_diff.py generated 19 anchored edits (+1 CMake wave64 edit) and verified they reproduce the fork file byte-for-byte and are idempotent. Per-route activation markers (patch=1256_nro07 path=topk_small/topk_parallel_radix). TOOLCHAIN BLOCKER: the fork's small-row route is compiled only for HIP >= 7.15; Brutus has ROCm 7.2.4 and 7.14, so only k==1 and ncols>1024 radix routes can activate on the fleet. Next: validation package (test-backend-ops TOP_K correctness both arms + markers; kernel perf via test-backend-ops perf mode; decode control).

OUTCOME 2026-09-27 (1256_nro07_topk_hybrid): series 1 (llama-bench MoE decode) never exercised TOP_K - PVPS10 profile showed MoE routing uses the fused topk_moe kernel (identical call counts both arms). Series 2 (contract amended before data): llama-server MTP decode with --backend-sampling, where request top-k runs ggml_top_k over the full vocabulary (pre-flight showed 1256's topk_small route). Activation passed all 8 sessions. Contract NRO07-TOPK-HYBRID: FAIL (not established) - point +1.18% gfx1100 (ci95_low -0.67), +1.45% gfx1201 (ci95_low -1.79); MTP lane too noisy at 10 rounds. Owner: move on (no series 3). Left untested, not rejected.

## Change Log

- 2026-09-09T10:52:38.138969+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:08:57.337897+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.075875+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.719952+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:42:40.812059+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria
- chg_20260910_024304_three-nasone-successors-now-pr_2691
- 2026-09-10T02:43:04.858380+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T02:26:26.803643+00:00 (updated-by): Updated: section:validation, section:notes
- 2026-09-24T04:49:23.947783+00:00 (updated-by): Updated: section:description, section:steps, section:notes
- 2026-09-24T15:40:54.324559+00:00 (state-transition): State: pending → in_progress
- 2026-09-24T15:40:57.228620+00:00 (updated-by): Updated: section:notes
- 2026-09-27T02:47:42.509365+00:00 (updated-by): Updated: section:notes
- 2026-09-27T02:47:57.300249+00:00 (state-transition): State: in_progress → pending
- 2026-10-10T10:25:19.318004+00:00 (state-transition): State: completed → completed
