# 1332_qsa_token_chunk

**Status:** validated
**Plan item:** QFP17

## What it does

With `BIGCHERRY_QSA_CHUNK=<tokens>` (e.g. 256), `build_qsa_sel` returns the remapped selection indices instead of
the dense `[n_kv + n_sel, T]` mask, and `build_attn_qsa` builds the same mask (fill -inf, scatter zeros, out-of-place
add of the causal rows) per chunk of query tokens and runs QSA flash attention per chunk, concatenating the outputs.
The 1331 peak trace showed the ub1024 compute-buffer peak is the two QSA masks (484 + 480 MiB) next to the kq_mask
input (480 MiB); chunking keeps one chunk's pair live (~240 MiB at 256 of 1024). Off by default. Conflicts with 1330.

## Result (b11402, on top of 1334)

`-ub 1024` with `BIGCHERRY_QSA_CHUNK=256` against `-ub 512`, production config with MTP: prefill +4% at ~99K and
~202K tokens, decode time per step unchanged, next-token fidelity within the envelope of the other attention-path
changes. Evidence and the promotion rationale are in README.md.
