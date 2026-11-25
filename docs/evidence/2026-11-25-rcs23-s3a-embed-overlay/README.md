# RCS23 S3a: the embedding gather as the first owned IQR overlay

Unit harness, `overlay-build` and `overlay-measure`, family `iqr_embed`, against release 2020.14.0, both architectures. The
overlay (`libbciqr`, one fat plugin for gfx1100 and gfx1201) is run over a minimal build of the same source whose embed launch
is a stub, so a launch that reached the base's kernel would fail: a result equal to the real base's means the overlay's own kernel
ran.

| check | gfx1201 | gfx1100 |
|---|---|---|
| kernels of the unit (`iqr_embed_kernel`, mangled names) with identical resource records (VGPR, SGPR, spills, private, LDS, workgroup) | 17 of 17 | 17 of 17 |
| selftest case `embed_lookup_iqr`: output hashes of 13 forms x 3 tables (512x640 at mult 1 and 3.25, 32768x5120) overlay vs base | 39 compared, 0 differ | 39 compared, 0 differ |
| the case's own oracle (host reference, bit for bit) | 0 failures | 0 failures |
| a plain bf16 table through the overlay row's layout hook | refused (-2, as the base) | refused (-2, as the base) |
| cold median launch time (256 MiB flush, 31 launches, best of 4 alternating rounds), ratio overlay / base, 13 forms on 32768x5120 | 0.994 to 1.006 (base 27.2 to 29.0 us) | 0.967 to 1.026 (base 24.5 to 30.9 us) |

The gather moves 9 token rows, so a launch is about 25 to 30 us on either card and the ratios are the card's launch-to-launch
noise; the check is "no regression", not a speed claim.

Row catalogue (tools/tests/release/test_bc_iqr_rows.py, on the staged libraries): the overlay declares exactly one row,
`bc_embed_lookup_iqr` (op `embed_lookup`, priority 60), constraints equal to the base row `embed_lookup_iqr`'s (priority 50);
with the overlay first the registry's rule picks it, with the base alone the base's rows serve the op, and the overlay declares
no op beyond the one it overrides. It cannot shadow another embedding: its layout hook is the base's IQR hook, which refuses
other encodings, and it refuses when the base cannot be resolved. libr3 no longer builds `r4d_iqr_embed` (its rewrite is archived
unchanged); every serving config whose kernels line contains `libr3` lists `libbciqr` first (test_bc_iqr_kernel_order.py).

Origin drift: `bigcherry pin-triage <ref>` reports each owned unit of `origins.toml` as clean, candidate (the origin path differs
between the last integrated commit and the target, with the commits) or blocked (a missing commit, a path absent at the last
integrated commit, a rename or removal, a target that is not a descendant, a shallow mirror). Against upstream main at the time of
this change the embed unit is clean.
