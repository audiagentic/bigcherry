# bc-iqr

## Summary

BigCherry's owned IQR kernels as one overlay plugin (`libbciqr`) for gfx1100 and gfx1201. It is listed ahead of libr3 and libr4d
in `--kernels`; a row of it wins where its constraints and the base's weight layout hook accept, and the base library
serves everything else.

## Outcome

First unit: the embedding gather over an IQR table (`bc_iqr_embed.hip`, row `bc_embed_lookup_iqr`). It is libr4d's
`r4d_iqr_embed.hip` with libr3's one rewrite block (the wave-shuffle Hadamard on gfx1100, guarded by `__gfx1100__`) part of
the text, so one source serves both architectures. libr3 no longer builds that unit (its rewrite is archived unchanged in
`libr3/native-shelved/`). The proof is the unit harness (`unit_harness.py overlay-build` and `overlay-measure`): output
hashes of the selftest case against the base row, resource records of the unit's kernels, a cold timing, on both
architectures.

## Detail

- `origins.toml` names each owned unit's upstream path, the commit it was taken from and the last commit integrated;
  `pin-triage` reports a drift of that path as `candidate` (changed upstream) or `blocked` (path or commit missing).
- Rows reuse the base's layout, relayout and shape hooks through the loaded base library (libr4d.so, else libr3.so), so the stored
  bytes are unchanged and a layout the base refuses is refused here (the row cannot shadow another weight form).
- A launch the owned kernel declines is handed back to the base's own launch function.
- Not in this component yet: the vocabulary head and the `r4d_gemm_iqr` family (RCS23 S3b, S3c).
