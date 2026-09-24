---
id: PNRO04
order: 0
plan: patching-nasone-rdna-optimizations
state: pending
created-at: '2026-09-09T10:52:17.766702+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: P0
---

# gfx1100 BF16/WMMA fused chunked GatedDeltaNet

## Description

Port and qualify gfx1100 BF16/WMMA chunked GatedDeltaNet on top of the existing RD50 recurrence. Keep BF16 operands with FP32 accumulation/state, exact gfx1100 geometry, and independent non-gfx11 fallback.

## Steps

1. Require existing RD50 chunked recurrence; extract only gfx11 BF16 WMMA implementation and helpers.
2. Gate host dispatch on gfx1100/RDNA3 and exact supported geometry; compile WMMA only in gfx11 device pass and never cross-select gfx12.
3. Add opt-out/force control and preserve sequential fallback.
4. Validate WMMA fragment layout with deterministic identity/structured/random/adversarial matrix probes.
5. Compare one/multi-chunk recurrence, recurrent state, CPU/reference outputs, long-sequence PPL/KL/greedy quality, graph/non-graph execution.
6. Profile KKT/scan separately and promote only gfx1100 exact shape after quality and E2E evidence.

## Detailed Solution & Technical Design

Capability owner: patching

Split assessment: One independent boundary; Build/Run support is a dependency.

Overlap assessment: No duplicate boundary found; related items are prerequisites or adjacent evidence.

## Code Samples & Guidance



## Files

patches/1253_nro04_gfx1100_bf16_chunked_gdn; GDN translation unit/helpers; shared package tests; WMMA/GDN fixtures

## Validation

Patch mechanics: `PYTHONPATH=tools python -m bigcherry patch-lint patches/1253_nro04_gfx1100_bf16_chunked_gdn`; `PYTHONPATH=tools python -m bigcherry patch-rebase-check --focal-overlay 1253_nro04_gfx1100_bf16_chunked_gdn --source bigcherry-tuning`; package pytest offline (static arch-guard/non-gfx11 compile-exclusion checks). Deterministic WMMA fragment matrix probe (identity/structured/random/adversarial). Hardware (Brutus, gfx1100 required): `python -m bigcherry.patch.validation_campaign --overlay 1253_nro04_gfx1100_bf16_chunked_gdn --arch gfx1100` comparing one/multi-chunk recurrence vs CPU/reference, long-sequence PPL/KL/greedy quality, graph/non-graph execution, KKT/scan profiling; gfx1201/gfx1030 must remain non-selecting controls.

## Effort & Risk



## Standards

Correctness before performance; architecture-specific promotion; source SHA fixed; no synthetic inheritance of RDNA4 evidence.

## Acceptance Criteria

gfx1100 WMMA primitive and GDN/state pass registered tolerances; non-gfx11 devices compile and cannot select; real gfx1100 effect is positive without quality regression; fallback remains verified.

## Notes

Supersedes: NRO04
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-nasone-rdna-optimizations-nro04

Supersedes: NRO04
Inherited semantic scope: preserve gfx1100 geometry, WMMA fragment probe, FP32 state, architecture separation, and fallback gates.
Migration: capability-rebaseline-v3-2026-09

2026-09-24 relevance at b11126: IMPLEMENTED-AS-PATCH. patches/1253_nro04_gfx1100_bf16_chunked_gdn exists, state=untested. Builds on existing RD50 chunked recurrence (already in bigcherry patch history). No upstream gfx1100 BF16/WMMA GDN equivalent found relevant to this project's patch history. Disposition: validate/qualify existing patch; no GPT design needed.

## Change Log

- 2026-09-09T10:52:17.766702+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:08:46.573893+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.067141+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.708292+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:26:15.401708+00:00 (updated-by): Updated: section:description, section:steps, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes
- chg_20260910_022800_five-nasone-successor-plans-no_4030
- 2026-09-10T02:28:00.297184+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T02:26:11.680244+00:00 (updated-by): Updated: section:validation, section:notes
