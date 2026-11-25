# rad0065_iqr_fused_gfx1201

State: **untested** until promoted.

## Summary

The gfx1201 prefill body reads its IQR weight fragments from block-local storage after each wave decodes one 16-row unit, instead of from a decoded plane in global scratch. The native WMMA instructions, integer accumulation and scale folds are those of the existing body. A constexpr certificate restricts the route to computed trellis kind 6 (L=16 B2 to B7 with the row-f32 scale, L=15 B2 row-f32, L=16 B4 to B7 bf16 with LRB=0 and LV=0), and dispatch requires exactly one 128-row band; every other call keeps the plane route.

## Outcome

Built on both architectures with the patch applied (experiment lane): libr3 (gfx1100) 7802 records, none added, changed or removed against the shipped release; libr4d (gfx1201) 7735 against 7724, 11 fused kernels added (VGPR 217 to 239, no spills, no private memory, 36864 B LDS), the 44 plane variants unchanged. On the R9700: 491 of 491 output hashes identical to the shipped plugin (stock tiled matrix plus M 65 to 2049 on the trunk shapes, B6 and B7, both scale forms), 1473 oracle checks pass. Cold replay, 60 launches a point, ABBA: all 24 single-band points (M 65, 96, 128) win, time 0.47 to 0.86 of shipped (13312x5120 at 128 rows 367 to 177 us); controls at 192, 256, 512 and 2048 rows unchanged.

## Detail

The tile and its dispatch live in libr4d/r4d_gemm_iqr.hip under distinct gfx12 identifiers (IqrFused12Pre, IqrFused12Certificate, iqr_fused12_prefill_kernel) and are compiled only where R3_TARGETS is undefined, so libr3 keeps its own gfx11 tile. The shared prefill header gets the same Fused template parameter and four body edits that libr3's native rewrite makes for gfx11; the libr3 rewrite treats those five blocks as optional, paired with an identity block that occurs when the patch has applied them, so a source with neither form still fails the rewrite check.

Mechanics: `python3 -m pytest tools/tests/patch/test_radiance_rad0065.py` and `PYTHONPATH=tools python3 -m bigcherry patch-lint`.
