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
- Second unit (S3b): the vocabulary-weighted head, `bc_iqr_vocab.hip` (row `bc_logits_gemm_vocab_iqr`): libr3's nine rewrite blocks
  are inlined under `#if defined(__gfx1100__)` with the origin's text in the `#else`, and libr3's `r3_compat.h` is included for the gfx1100
  device pass. The scratch hook is the unit's own (it lived in the unit libr3 no longer builds).
- Third unit (S3c): the IQR GEMM family, `bc_iqr_gemm.hip` (+ parts 1 to 3) with rows `bc_gemm_iqr_nt_m64`, `bc_gemm_iqr_gated`,
  `bc_gemm_iqr_tiled`, including the single-band fused prefill tile (the text of the retired patch rad0065). It includes an owned
  extract of the shared 8a8 prefill header, `bc_gemm_8a8_prefill.h` (upstream's text with rad0061's and rad0065's header edits, plus
  libr3's gfx11 blocks; strategy extracted-family in `origins.toml`). Kernel bodies differ per target under `__gfx1100__`; the host
  half is compiled once, so the two routes only gfx1100 had (the rotated 4-bit table slice route and its scratch rule) are chosen at
  run time from the card. The three host helpers the source asks of upstream's fp8a8 unit (split-K counter, narrow tile geometry)
  are forwarded to the base of the card; a launch without them declines before any GPU work. No hand-back to the base for these rows:
  the source is the base's own code, so a form it declines the base declines too.
