---
id: PNRO07
order: 0
plan: patching-nasone-rdna-optimizations
state: completed
created-at: '2026-09-09T10:52:42.526873+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: P0
---

# Wave32-native TOP_K reductions and item-per-thread tuning

## Description

Disposition (2026-10-08): **closed without promotion** as a dependent increment to PNRO06/1256. The wave32 1257 port is real and independently activated in the historical 2026-09-27 campaign, but did not establish an E2E gain on gfx1100 or gfx1201. It inherits 1256's conflict with validated 1294 deterministic QSA ties. Do not resume a third hardware series.

## Steps

1. Retain 1257's untested patch and exact fork provenance as historical evidence; do not add to production or independently rebase the kernels.
2. PNRO06 owns any future generic TOP_K selector/correctness decision; 1294/RNX02 own QSA deterministic ordering. Wave32 tuning cannot bypass that dependency.
3. Reopen only if PNRO06's new production TOP_K callsite passes the >=5% E2E attribution gate, deterministic tie parity and architecture/toolchain eligibility. Compare native vs 1256 vs 1256+1257 in one controlled matrix; otherwise terminate without implementation.

## Detailed Solution & Technical Design

2026-10-08 audit: `patches/1257_nro08_topk_wave32/patch.py` is an exact nasone `7f1d25f7` follow-up to 1256, with two-half 64-bin wave32 scans, shuffle/LDS changes and removal of the wave64 compile flag. It does **not** change the caller: `ggml_cuda_op_top_k()` remains the only generic selection owner. Pinned b11474 and current upstream have identical native TOP_K code. The 1257 package requires 1256, which conflicts with validated `1294_topk_deterministic_ties`; a wave32 microkernel cannot solve the QSA tie/order contract. The 2026-09-27 series-2 incremental measured effects were +0.87% (gfx1100, CI95-low -1.17%) and +0.51% (gfx1201, CI95-low -3.09%); neither qualifies. The production MoE series used fused `topk_moe` instead of these kernels. Preserve gfx1030 and non-HIP fallback. If a successor ever qualifies, require 31/32/33, 63/64/65 boundaries, ties, multi-ubatch, graph replay, bit/greedy parity, LDS/VGPR/barrier counts, >=4 sessions and CI95-low >=3% E2E with <=1% control regression. No independent wave32 scheduler or config surface.

## Code Samples & Guidance



## Files

patches/1257_nro08_topk_wave32/{patch.toml,patch.py,SUMMARY.md,README.md,TESTING.md}; PNRO06 fixtures; wave32 boundary tests; profiler/campaign evidence.

## Validation

Existing 2026-09-27 hardware evidence reviewed, source/metadata and dependency checks performed; no new HIP compile or hardware benchmark. The 48-case host route fixture covers 1256 dispatch, not 1257 numerical behavior. Retain the patch's historical `untested` state and close this optimisation plan without promotion.

## Effort & Risk



## Standards

Dependency-aware composition; exact routing semantics; preserve fallback; no unsupported wave-size assumptions.

## Acceptance Criteria

Terminal: no 1257 promotion, no duplicate selector, no new queued campaign. Any later wave32 successor is subordinate to a newly qualified PNRO06 signature and deterministic 1294-compatible ordering.

## Notes

Supersedes: NRO08
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-nasone-rdna-optimizations-nro08

2026-09-24 relevance at b11126: IMPLEMENTED-AS-PATCH. Item title is wave32 TOP_K follow-up; its own Files section and patches/ dir map it to patches/1257_nro08_topk_wave32 (state=untested, requires 1256_nro07_topk_hybrid i.e. PNRO06's patch) -- package literally named nro08 (matches its "Successor key: nro08"/Supersedes NRO08), verified via patches/1257*/patch.toml id field, consistent with the plan item id PNRO07. Disposition: validate/qualify existing patch; no GPT design needed.

2026-09-24 GPT review req_215c89d0b13a4bb7 applied: verified 1257 only adds an unused TOP-1 wave32 reduction helper with no scan/ITEMS_PER_THREAD wiring and no caller. Rescoped as blocked on PNRO06's rebase (its target kernels don't have a stable post-rebase identity yet) and required the wave32 changes be applied to the real, concrete kernels with a distinct activation marker, rather than left as dead code.

2026-09-25 (ef49e4e5): 1257 is now an exact port of nasone 7f1d25f7 (wave32-native TOP_K) on top of 1256, port_diff-generated (16 edits + CMake flag removal), verified byte-exact; markers tagged patch=1257_nro08. No longer blocked on a PNRO06 'rebase' -- 1256 is the exact pre-image. Same HIP >= 7.15 toolchain caveat as PNRO06.

OUTCOME 2026-09-27 (1257_nro08_topk_wave32, on top of 1256): series 2 backend-sampling MTP decode, activation passed all 8 sessions. Contract NRO08-TOPK-WAVE32: FAIL (not established) - point +0.87% gfx1100 (ci95_low -1.17), +0.51% gfx1201 (ci95_low -3.09). Owner: move on. Left untested, not rejected.

## Change Log

- 2026-09-09T10:52:42.526873+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:09:02.680624+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.081564+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.725809+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:42:47.187884+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria
- chg_20260910_024304_three-nasone-successors-now-pr_2691
- 2026-09-10T02:43:04.870594+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T02:26:33.904175+00:00 (updated-by): Updated: section:validation, section:notes
- 2026-09-24T04:49:31.554021+00:00 (updated-by): Updated: section:description, section:steps, section:notes
- 2026-09-24T15:41:00.121413+00:00 (state-transition): State: pending → in_progress
- 2026-09-24T15:41:03.007035+00:00 (updated-by): Updated: section:notes
- 2026-09-27T02:47:45.532222+00:00 (updated-by): Updated: section:notes
- 2026-09-27T02:48:00.326472+00:00 (state-transition): State: in_progress → pending
- 2026-10-10T10:25:25.053304+00:00 (state-transition): State: completed → completed
