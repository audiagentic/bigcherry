# 1270_pnro14_rdna35_fa_tile_d256

**Status:** untested
**Plan item:** PNRO14

## What it does

Adds one gfx1151/RDNA3.5 tile-FlashAttention configuration for `(DKQ,DV,ncols)=(256,256,32)`: 256 threads, occupancy target 4, `nbatch_fa=64`, `nbatch_K=64`. Host and device selectors use it only on RDNA3.5; every other shape/architecture falls through to the existing b11126 RDNA tables.

Qualification is hardware-deferred until gfx1151 is available. Promotion requires backend-reference correctness, activation of the exact row, an established prefill improvement, and <=1% decode regression.
