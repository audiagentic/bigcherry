# ngram-helper: n-gram embedding gather on a spare GPU

## Summary

Opt-in n-gram row gather on a non-rank card (`R3_NGRAM_DEV`), split out of
`libr3` (RCS06).

## Outcome

`evaluated` (shelved): exact, but default off and outside the production set.
Owner decision 2026-10-13 — the current n-gram-on-R9700 topology is a poor
use of that card; the R9700 belongs to compute (the moe-remote routed-expert
helper), and lookup-type side work like this belongs on host RAM/CPU instead.
Kept on main per the "shelved work is kept, never deleted" rule in case the
topology question is revisited.

## Detail

All settings under this component (`R3_NGRAM_DEV`, `R3_NGRAM_HELPER_SO`,
`R3_NGRAM_AVX`, `R3_NGRAM_RAM`, `R3_NGRAM_MIB`) are `evaluated` — see
`component.toml` for each one's summary and activation marker. To use it
anyway for further investigation: list `libr3ngram` in `--kernels` ahead of the
engine's kernel library and set `R3_NGRAM_DEV`. Do not spend new measurement
runs on R9700-as-storage topologies per the owner's 2026-10-13 call.
