# 1270_pnro14_rdna35_fa_tile_d256

gfx1151-only D=256 tile FlashAttention occupancy experiment from PNRO14. The patch adds one `(256,256,32)` row and preserves the existing RDNA table as the fallback for all non-target shapes and architectures.

Activation marker:
`BIGCHERRY_PATCH_HIT patch=1270_pnro14_rdna35_fa_tile_d256 path=tile_d256_cols32 contract=PNRO14-RDNA35-FA-TILE-D256`

Current fleet has no gfx1151; hardware qualification is intentionally deferred rather than extrapolated from gfx1100/gfx1201.
