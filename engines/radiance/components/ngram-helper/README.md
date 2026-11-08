# ngram-helper: n-gram embedding gather on a spare GPU

## Summary

Opt-in n-gram row gather on a non-rank card (once `R3_NGRAM_DEV`; now off), split out of
`libr3` (RCS06).

## Outcome

`evaluated` (shelved): exact, but default off and outside the production set.
Owner decision 2026-10-13 — the current n-gram-on-R9700 topology is a poor
use of that card; the R9700 belongs to compute (the moe-remote routed-expert
helper), and lookup-type side work like this belongs on host RAM/CPU instead.
Kept on main per the "shelved work is kept, never deleted" rule in case the
topology question is revisited.

## Detail

The component's environment settings (`R3_NGRAM_DEV`, `R3_NGRAM_HELPER_SO`, `R3_NGRAM_AVX`, `R3_NGRAM_RAM`,
`R3_NGRAM_MIB`, `R3_NGRAM_RESERVE_MIB`, `R3_NGRAM_MAX_M`) are gone (RSC12: no environment settings): the gather is off, and
the helper libraries are found next to the plugin. It is not a manifest field because nothing uses it and the owner
shelved it (2026-10-13); if the topology question is revisited, selecting a card becomes a field under
`spec.processes.engine`, bound like the kernel switches. History: `docs/archive/radiance/LAB_HISTORY.md`.
