# RCS23 S3c: the IQR GEMM family as the third owned unit; rad0065 retired

Unit harness, `overlay-build` and `overlay-measure`, overlay `iqr_gemm` (selftest case `gemm_iqr_tiled`), against release 2020.14.1,
both architectures. The overlay runs over a minimal base that has the int8 reference rows and the fp8a8 helpers the unit asks for but
not the IQR units, so a hash equal to the real base's means the overlay's own kernel ran. The owned source is `bc_iqr_gemm.hip` (four
parts) with `bc_gemm_8a8_prefill.h` (the shared header, extracted); the single-band fused tile in it is the text of the retired patch
rad0065, so it exists once. Kernel bodies differ per target under `__gfx1100__`; the two host-visible routes only gfx1100 had are
chosen at run time from the card, because the fat plugin has one host object.

| check | gfx1201 | gfx1100 |
|---|---|---|
| kernel resource records of the whole overlay (embed, head, GEMM family: 3203 kernels) against the release | 3203 of 3203 identical | 3198 identical; the 5 others are the known cold-build forms, which the overlay compiles to the earlier 236 VGPR, 0 spill records the release's libr3 had lost (release: 240 VGPR with 2 or 3 spills) |
| output hashes, overlay vs base: the case's own forms plus the trunk shapes at 65 to 2049 rows, tails and a ragged N (555 cases) | 555 compared, 0 differ | 555 compared, 0 differ |
| the case's oracle (every form against the int8 reference row) | 0 failures | 0 failures |
| a plain bf16 table through the rows' layout hook | refused (-2, as the base) | refused (-2, as the base) |
| cold replay (256 MiB flush, ABBA windows in separate processes), 60 points: 3 trunk shapes x rows 1, 16, 64, 65, 96, 128, 129, 256, 512, 2048 x 4-bit and 7-bit forms | 60 same (0.991 to 1.009) | 54 same, 3 win (0.73 to 0.94), 3 flagged and rerun: the three points sit on a bimodal launch (317 or 446 us at 65 and 128 rows, 50 or 60 us at one row on 13312x5120, 4-bit) that either plugin takes depending on the process; the rerun gave 1.02, 1.00, 1.01, 0.995 |

Compose gates on the new production set (rad0065 out of it, libr3 no longer building the unit): gate 1 passes; gate 2a, 444 syntax
passes (host and device, both architectures), clean.

The patch is archived unchanged (`patches/archive/rad0065_iqr_fused_gfx1201`, ARCHIVE.md row); libr3 excludes `r4d_gemm_iqr` and its
three parts and keeps its header rewrite for the int8 rows (whose optional blocks for the retired patch's text simply do not occur);
the rewrite of the unit is shelved unchanged. Every serving config whose kernels line has `libr3` or whose model is an IQR container
(iq4r, iq6r) lists `libbciqr` first, tested together with the no-orphan-overlay rule.
