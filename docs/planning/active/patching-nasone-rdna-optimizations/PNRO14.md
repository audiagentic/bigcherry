---
id: PNRO14
order: 0
plan: patching-nasone-rdna-optimizations
state: pending
created-at: '2026-09-09T10:53:13.365873+00:00'
breadth: ''
skill: intermediate
created-by: capability-rebaseline-v3
work: S
priority: P2
---

# RDNA3.5 D=256 tile FlashAttention occupancy configuration

## Description

Retain the RDNA3.5 D=256 FlashAttention occupancy row as a future-hardware candidate; current gfx1100/gfx1201 must not select it.

## Steps

- Check upstream landing when gfx1151 hardware/source is available.
- If absent, port only GGML_CUDA_FATTN_TILE_CONFIG_CASE(256,256,32,256,4,64,64) and RDNA3.5 host/device selection.
- Build and run non-selection controls on gfx1100/gfx1201.
- On gfx1151 with rocWMMA disabled, measure D=256/ncols=32 across prompt lengths with correctness, LDS/VGPR/occupancy evidence.
- Promote only on positive gfx1151 evidence; similar geometry on other architectures requires separate qualification.

## Detailed Solution & Technical Design

Use a distinct RDNA3.5 config function delegating other shapes to the generic table; no graph-opt or cross-architecture default rides along.

## Code Samples & Guidance



## Files

FA tile config source/table; host/device RDNA3.5 selector; build/non-selection tests; gfx1151 campaign/profiler evidence.

## Validation

Upstream/patch ancestry; gfx1100/gfx1201 non-selection; gfx1151 D=256/ncols=32 correctness/perf; LDS/VGPR/occupancy.

## Effort & Risk



## Standards

Architecture-scoped evidence; no extrapolation.

## Acceptance Criteria

Correct host/device table selection; no non-RDNA3.5 selection; positive gfx1151 prefill effect with unchanged correctness, otherwise retain deferred status.

## Notes

Supersedes: NRO15
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-nasone-rdna-optimizations-nro15

## Change Log

- 2026-09-09T10:53:13.365873+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:09:43.442305+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.111896+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.772814+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:46:10.318015+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria
- chg_20260910_024630_the-remaining-nasone-successor_5195
- 2026-09-10T02:46:30.073220+00:00 (updated-by): Updated: section:ledger-events
