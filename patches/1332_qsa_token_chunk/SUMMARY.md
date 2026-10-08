# 1332_qsa_token_chunk

**Status:** validated
**Plan item:** QFP17

## What it does

With `BIGCHERRY_QSA_CHUNK=<tokens>` (e.g. 256), batches with more than 8 tokens apply the causal predicate while
the selection is still compact, return remapped I32 selection indices instead of the dense
`[n_kv + n_sel, T]` mask, and build one final dense selection mask per query-token chunk before flash attention.
There is no dense causal-mask add on this chunked path. Decode/MTP-sized batches stay on the original dense fallback.
The 1331 peak trace identified the unchunked QSA masks as the ub1024 compute-buffer peak driver. Off by default.

1330 is composable after 1332: it only optimizes the dense fallback that 1332 deliberately retains.

## Result (b11402, on top of 1334)

`-ub 1024` with `BIGCHERRY_QSA_CHUNK=256` against `-ub 512`, production config with MTP: prefill +4% at ~99K and
~202K tokens, decode time per step unchanged, next-token fidelity within the envelope of the other attention-path
changes. Evidence and the promotion rationale are in README.md.
