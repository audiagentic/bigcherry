# 1270_pnro14_rdna35_fa_tile_d256

**Status:** untested / deferred; PNRO14 completed as no-qualified-gfx1151-lane (2026-10-08), NOT promoted
**Plan item:** PNRO14

## What it does

Adds one gfx1151/RDNA3.5 tile-FlashAttention configuration for `(DKQ,DV,ncols)=(256,256,32)`: 256 threads, occupancy target 4, `nbatch_fa=64`, `nbatch_K=64`. Host and device selectors use it only on RDNA3.5; every other shape/architecture falls through to the existing b11126 RDNA tables.

Qualification is hardware-deferred until gfx1151 is available. The host marker is emitted during tile configuration lookup, not after a kernel launch; it cannot by itself prove the target TILE path executed. Upstream MMA is selected first for D256 with eligible GQA and sufficient effective batch, potentially bypassing this row. Promotion requires actual TILE launch attribution, >=5% target E2E wall-time share, backend-reference correctness, >=4 sessions, CI95-low >=3% E2E improvement, and <=1% control regression. Do not infer benefit from gfx1100/gfx1201 or external gfx1151 MMA-vs-TILE measurements. Patch anchor failures described by PNRO19 were fixed by f422854 on 2026-09-27.
