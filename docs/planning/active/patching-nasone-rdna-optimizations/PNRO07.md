---
id: PNRO07
order: 0
plan: patching-nasone-rdna-optimizations
state: pending
created-at: '2026-09-09T10:52:42.526873+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: P0
---

# Wave32-native TOP_K reductions and item-per-thread tuning

## Description

Apply and qualify the wave32-native TOP_K follow-up only as a causal increment on PNRO06.

## Steps

- Require PNRO06 post-image and verify source pre-image before applying 7f1d25f7...
- Port explicit 32-lane shuffle/fallback structure, two-half 64-bin scans, and ITEMS_PER_THREAD tuning separately where possible.
- Run PNRO06 correctness fixtures unchanged plus 31/32/33 and 63/64/65 boundaries, ties, NaNs and block/item coverage edges.
- Profile LDS traffic, occupancy, VGPRs, pass count and duration; compare PNRO06-only control with PNRO06+PNRO07 on identical real signatures.
- Retain selector/fallback when wave32 assumptions are not proven or benefit is subset-only; do not extrapolate wave size.

## Detailed Solution & Technical Design

Wave32 reductions must use explicit width/masks and keep shared-memory fallback. Two-half radix scans must validate threshold selection around the 32/64 boundary. Separate semantic and items/thread changes for attribution.

## Code Samples & Guidance



## Files

patches/1257_nro08_topk_wave32/{patch.toml,patch.py,SUMMARY.md,README.md,TESTING.md}; PNRO06 fixtures; wave32 boundary tests; profiler/campaign evidence.

## Validation

Patch mechanics: `PYTHONPATH=tools python -m bigcherry patch-lint patches/1257_nro08_topk_wave32`; `PYTHONPATH=tools python -m bigcherry patch-rebase-check --focal-overlay 1257_nro08_topk_wave32 --source bigcherry-tuning --requires 1256_nro07_topk_hybrid`; package pytest offline. PNRO06(nro07) correctness fixtures unchanged plus 31/32/33 and 63/64/65 boundaries, ties, NaNs, block/item coverage edges. Hardware (Brutus, gfx1100, wave32-capable): `python -m bigcherry.patch.validation_campaign --overlay 1257_nro08_topk_wave32 --requires 1256_nro07_topk_hybrid --arch gfx1100` profiling LDS traffic/occupancy/VGPRs/pass count/duration, PNRO06-only control vs PNRO06+PNRO07 on identical real signatures; <=1% non-target regression gate.

## Effort & Risk



## Standards

Dependency-aware composition; exact routing semantics; preserve fallback; no unsupported wave-size assumptions.

## Acceptance Criteria

Outputs match PNRO06/reference; wave32 activates only where supported; real gfx1100 signatures show measurable kernel/LDS/pass benefit; no >1% non-target regression; otherwise retain PNRO06 fallback.

## Notes

Supersedes: NRO08
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-nasone-rdna-optimizations-nro08

2026-09-24 relevance at b11126: IMPLEMENTED-AS-PATCH. Item title is wave32 TOP_K follow-up; its own Files section and patches/ dir map it to patches/1257_nro08_topk_wave32 (state=untested, requires 1256_nro07_topk_hybrid i.e. PNRO06's patch) -- package literally named nro08 (matches its "Successor key: nro08"/Supersedes NRO08), verified via patches/1257*/patch.toml id field, consistent with the plan item id PNRO07. Disposition: validate/qualify existing patch; no GPT design needed.

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
