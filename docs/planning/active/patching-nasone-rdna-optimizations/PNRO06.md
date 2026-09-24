---
id: PNRO06
order: 0
plan: patching-nasone-rdna-optimizations
state: in_progress
created-at: '2026-09-09T10:52:38.138969+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: P0
---

# Hybrid ROCm TOP_K kernels

## Description

TODO, NOT-READY (upstream premise stale, per GPT review). Port and qualify nasone hybrid HIP TOP_K selection -- CORRECTED: the plan's upstream premise is stale. b11126's ggml/src/ggml-cuda/top-k.cu already has a real HIP `top_k_radix_cuda()` path for ncols>1024 (top-k.cu:180, called at top-k.cu:263) plus a bitonic fallback for smaller shapes, including `top_k_float_to_ordered()` (top-k.cu:53) -- this is NOT a from-scratch port target; the real remaining scope is identifying what, if anything, source 7f3e1e4d... adds beyond what b11126 already has natively.

## Steps

1. Re-diff source 7f3e1e4d... against the CURRENT b11126 top-k.cu (which already has top_k_radix_cuda for ncols>1024 and a bitonic fallback, verified) -- port only genuinely missing TOP-1/n-ary selection paths, not the whole file.
2. Wire any genuinely-missing path in ggml_cuda_op_top_k() (verify exact function name) AHEAD of the existing HIP radix/bitonic routes, under an explicit opt-in gate.
3. Retain the current radix/bitonic implementation as the fallback in all other cases.
4. Add an actual-dispatch activation marker (BIGCHERRY_PATCH_TRACE-gated) so the new path's selection is observable.
5. Build CPU/reference fixtures for k/nrows/ncols, ties, negative values, infinities and duplicates; verify caller ordering semantics against BOTH the existing radix/bitonic path and any newly-ported path.
6. Capture real MoE/QSA signatures and compare hybrid against the existing (not invented) bitonic/radix baseline at k=1 and routing k=2/4/8/10; keep PNRO07 disabled for causal attribution.

## Detailed Solution & Technical Design

The optimization changes algorithmic complexity and temporary storage, not just geometry. Exact index/tie semantics are load-bearing; any stable-order divergence requires an explicit policy. Keep HIP-only containment and native fallback.

## Code Samples & Guidance



## Files

patches/1256_nro07_topk_hybrid/{patch.toml,patch.py,SUMMARY.md,README.md,TESTING.md}; top-k.cu; shared NRO static tests; CPU/reference TOP_K fixture runner; real signature campaign artifacts.

## Validation

Patch mechanics: `PYTHONPATH=tools python -m bigcherry patch-lint patches/1256_nro07_topk_hybrid`; `PYTHONPATH=tools python -m bigcherry patch-rebase-check --focal-overlay 1256_nro07_topk_hybrid --source bigcherry-tuning`; package pytest offline (non-HIP preprocessor preservation, idempotent apply). CPU/reference TOP_K fixtures: exact index/order for k=1..ncols, multi-row, ties, negatives, infinities, duplicates; unsupported-shape fallback to bitonic. Hardware (Brutus, gfx1100): `python -m bigcherry.patch.validation_campaign --overlay 1256_nro07_topk_hybrid --arch gfx1100` against real MoE/QSA signatures at k=1 and routing k=2/4/8/10, kernel/scratch/end-to-end call-weighted performance; keep PNRO07(nro08 wave32) disabled for causal attribution.

## Effort & Risk



## Standards

Exact routing correctness before performance; HIP-only containment; native fallback; no conflation with downstream routing fusion.

## Acceptance Criteria

Reference and routing correctness pass; non-HIP builds unchanged; unsupported shapes fall back; at least one real gfx1100 high-cost signature shows repeatable win without model regression; PNRO07 remains separate.

## Notes

Supersedes: NRO07
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-nasone-rdna-optimizations-nro07

2026-09-24 relevance at b11126: IMPLEMENTED-AS-PATCH. Item title says TOP_K hybrid; its own Files section and patches/ dir map it to patches/1256_nro07_topk_hybrid (state=untested) -- note the patch package is literally named nro07 (matches its "Successor key: nro07"/Supersedes NRO07 in Notes) even though the plan item id is PNRO06; this is the correct, verified mapping (grep patches/1256*/patch.toml id field confirms). No upstream HIP-native TOP_K hybrid selection kernel found in b11126 top-k.cu relevant to this scope. Disposition: validate/qualify existing patch; no GPT design needed.

2026-09-24 GPT review req_215c89d0b13a4bb7 applied: verified via grep that b11126 top-k.cu already has HIP top_k_radix_cuda (ncols>1024) and bitonic fallback with top_k_float_to_ordered -- corrected the plan's stale premise that this needed a from-scratch HIP port. Rescoped to a re-diff against the current source to find genuinely missing paths, wired ahead of the existing radix/bitonic routes under an opt-in gate, with a required dispatch marker.

2026-09-25 (ef49e4e5): the scaffold is replaced by an EXACT port of nasone 7f3e1e4d + 10fdba9a. The fork's pre-change top-k.cu is byte-identical to b11126's, so no rebase design was needed: new tools/bigcherry/patch/port_diff.py generated 19 anchored edits (+1 CMake wave64 edit) and verified they reproduce the fork file byte-for-byte and are idempotent. Per-route activation markers (patch=1256_nro07 path=topk_small/topk_parallel_radix). TOOLCHAIN BLOCKER: the fork's small-row route is compiled only for HIP >= 7.15; Brutus has ROCm 7.2.4 and 7.14, so only k==1 and ncols>1024 radix routes can activate on the fleet. Next: validation package (test-backend-ops TOP_K correctness both arms + markers; kernel perf via test-backend-ops perf mode; decode control).

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
