# rad0065_iqr_fused_gfx1201

State: **untested** until promoted.

## Summary

The gfx1201 prefill body reads its IQR weight fragments from block-local storage after each wave decodes one 16-row unit, instead of from a decoded plane in global scratch. The native WMMA instructions, integer accumulation and scale folds are those of the existing body. A constexpr certificate restricts the route to computed trellis kind 6 (L=16 B2 to B7 with the row-f32 scale, L=15 B2 row-f32, L=16 B4 to B7 bf16 with LRB=0 and LV=0), and dispatch requires exactly one 128-row band; every other call keeps the plane route.

## Outcome

Built on both architectures with the patch applied (experiment lane): libr3 (gfx1100) 7802 records, none added, changed or removed against the shipped release; libr4d (gfx1201) 7735 against 7724, 11 fused kernels added (VGPR 217 to 239, no spills, no private memory, 36864 B LDS), the 44 plane variants unchanged. On the R9700: 491 of 491 output hashes identical to the shipped plugin (stock tiled matrix plus M 65 to 2049 on the trunk shapes, B6 and B7, both scale forms), 1473 oracle checks pass. Cold replay, 60 launches a point, ABBA: all 24 single-band points (M 65, 96, 128) win, time 0.47 to 0.86 of shipped (13312x5120 at 128 rows 367 to 177 us); controls at 192, 256, 512 and 2048 rows unchanged.

## Detail

The tile and its dispatch live in libr4d/r4d_gemm_iqr.hip under distinct gfx12 identifiers (IqrFused12Pre, IqrFused12Certificate, iqr_fused12_prefill_kernel) and are compiled only where R3_TARGETS is undefined, so libr3 keeps its own gfx11 tile. The shared prefill header gets the same Fused template parameter and four body edits that libr3's native rewrite makes for gfx11; the libr3 rewrite treats those five blocks as optional, paired with an identity block that occurs when the patch has applied them, so a source with neither form still fails the rewrite check.

Mechanics: `python3 -m pytest tools/tests/patch/test_radiance_rad0065.py` and `PYTHONPATH=tools python3 -m bigcherry patch-lint`.

### Promotion record

Tier: profile-evidence

Model: Qwen3.8-27B IQR-V4 container (6/7-bit computed-trellis trunk), tensor parallel over two RX 7900 XTX and an R9700 (TP3); the patch acts on the R9700 (gfx1201) prefill GEMMs of the trunk.
Mechanics: tools/tests/patch/test_radiance_rad0065.py passes (applies once, idempotent, fails closed on a missing or duplicate anchor, routing guard); patch-lint is clean; the composition check (patch-compose-check) passes gate 1 (native rewrite anchors) and gates 2a and 2b (syntax-only and real compile of the five libr4d units the patch touches, host and device passes, gfx1201 and gfx1100: 20 passes each); the lane build carrying it (experiment on radiance:stock:linux-radiance) builds both architectures: libr3 (gfx1100) 7802 resource records, none added, changed or removed against release 2020.11.4, libr4d (gfx1201) 7724 to 7735 records, 11 fused kernels added (VGPR 217 to 239, no spills, no private memory, 36864 B of LDS), the 44 plane-route variants unchanged.
Activation: on the R9700 the fused kernel serves exactly the single-band prefill rows (up to 128 rows) of the 11 computed-trellis forms and nothing else: a cold replay of the IQR row on every trunk shape shows the time change only at 65, 96 and 128 rows (24 of 24 points) and none at 192, 256, 512 or 2048 rows, where the plane route still runs; the libr3 plugin of the same build is unchanged and does not contain the kernel.
Identity: output bytes identical to the shipped plugin: 491 of 491 output hashes over the stock tiled matrix plus 65 to 2049 rows on five trunk shapes (including N = 5376), 6-bit and 7-bit, bf16 and f32 scale forms, one R9700, and the 1473 oracle checks of the same matrix pass on both plugins.
A/B: cold replay (256 MiB flush before each launch, 60 launches a point, shipped and patched plugins ABBA, one R9700): all 24 single-band points win; time of the patched row against shipped 0.47 to 0.86 (13312x5120 at 128 rows 367 to 177 us, 5120x6656 165 to 129 us, 6144x5120 144 to 105 us, 5120x2304 63 to 55 us); the controls at 192, 256, 512 and 2048 rows are unchanged within 1.01; IQR against the int8 tiled route at 65 to 128 rows 2.5 times before, 1.75 times after. No serving A/B yet: the effect is on the first-token time of prompts of 65 to 128 tokens on the R9700 rank.
