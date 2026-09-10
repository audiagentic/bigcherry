---
id: NRO15
order: 15
plan: nasone-rdna-optimizations
state: superseded
created-at: '2026-09-08T09:50:40+10:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P2
work: S
---

# RDNA3.5 D=256 tile FlashAttention occupancy configuration

## Description

Retain nasone commit `ed11a0d2fc0f79b6a45833f79ffa36e3afe253b9` as a future-hardware candidate. With rocWMMA FA disabled, D=256/ncols=32 uses the tile kernel; the source adds an RDNA3.5-specific row reducing K tile 128->64 and raising occupancy target 3->4, falling back to the shared RDNA table for other cases.

This does not target RX 7900 XTX/gfx1100 and should not consume current Brutus GPU time except non-selection/build checks.

## Steps

1. Check whether the config has landed upstream by the time gfx1151 hardware is available.
2. If absent, port only the configuration row and host/device RDNA3.5 selection.
3. Verify gfx1100/gfx1201 do not select it.
4. On gfx1151, benchmark D=256,ncols=32 across prompt lengths with rocWMMA disabled.
5. Record LDS/VGPR/occupancy to verify the proposed mechanism.

## Detailed Solution & Technical Design

Exact source candidate: `GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256, 32, 256, 4, 64, 64)`. It should live in a distinct RDNA3.5 config function that delegates all other shapes to the generic RDNA table.

## Code Samples & Guidance

Do not generalize the row to gfx1100. A similar geometry on another architecture requires its own sweep/evidence.

## Files

Planning-only; future small FA config patch and table-selection tests.

## Validation

Build/non-selection on existing hardware; real gfx1151 correctness/performance before promotion.

## Effort & Risk

Low code risk, hardware-blocked performance evidence.

## Standards

Architecture-scoped evidence; no extrapolation.

## Acceptance Criteria

- Correct host/device table selection.
- No selection on non-RDNA3.5.
- Positive gfx1151 D=256 prefill effect with unchanged correctness.

## Notes

Retain rather than discard solely because current production is gfx1100.

Superseded by: PNRO14
Migration: capability-rebaseline-v3-2026-09

## Change Log

- 2026-09-08T09:50:40+10:00 (created-by): Created as future RDNA3.5 candidate; P2.

## Ledger-events



- Pending: ag-ledger MCP unavailable in authoring session.
- 2026-09-09T11:25:35.782944+00:00 (updated-by): Updated: section:notes
- 2026-09-09T11:43:40.492819+00:00 (state-transition): State: pending → superseded
- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:57:59.916128+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:37.053250+00:00 (updated-by): Updated: section:ledger-events
