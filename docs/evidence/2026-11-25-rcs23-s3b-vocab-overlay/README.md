# RCS23 S3b: the vocabulary-weighted IQR head as the second owned unit

Unit harness, `overlay-build` and `overlay-measure`, overlay `iqr_vocab` (selftest case `logits_gemm_vocab_iqr`), against release
2020.14.0, both architectures; the overlay runs over a minimal base whose launch for this row is a stub, so a result equal to the
real base's means the overlay's own kernel ran. The owned source `bc_iqr_vocab.hip` is libr4d's `r4d_logits_vocab.hip` with libr3's
nine rewrite blocks inlined under `#if defined(__gfx1100__)` (the origin's text in the `#else`) and `r3_compat.h` included for the
gfx1100 device pass: one source, both architectures.

| check | gfx1201 | gfx1100 |
|---|---|---|
| kernels of the unit with identical resource records | 6 of 6 | 6 of 6 |
| the case's logits hashed, overlay vs base (11 head shapes: the case's own plus 8x5120 over 4096 rows and 64x5120 over 2048 rows) | 11 compared, 0 differ | 11 compared, 0 differ |
| the case's oracle (f64 host product within 2^-18 of the rms; every logit written; planes invert) | 0 failures | 0 failures |
| a plain bf16 table through the row's layout hook | refused (-2, as the base) | refused (-2, as the base) |
| cold median launch time, ratio overlay / base (256 MiB flush, 31 launches, best of 8 rounds gfx1201 / 4 rounds gfx1100) | 1.00 on 9 of 11 points; two points are bimodal in both plugins (M=64 and M=100 on the small multi-class head: 49 or 86 us, 77 or 139 us, the fast or the slow mode depending on the process, in either arm) | 0.978 to 1.046 |

The bimodality is the same on the base alone (the first run had the slow mode in the overlay arm at one point and in the base arm at
another), so it is a property of the kernel's launch (scratch placement), not of the overlay; the hash and resource records are the
stable checks.

libr3 no longer builds `r4d_logits_vocab` (its rewrite is archived unchanged in `native-shelved/`; the head's scratch hook lived in
that unit and is now the overlay's own, the origin's code under an owned name). `origins.toml` lists the unit, so pin-triage reports
upstream changes to the origin file against the commit it was taken from.
