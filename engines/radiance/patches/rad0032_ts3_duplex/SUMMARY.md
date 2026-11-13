# rad0032_ts3_duplex

**Status:** untested
**Plan item:** VTP36

Scope: the exact two-shot all-reduce of a three-rank world whose links differ, under new `--tune-pin` axes of the
all-reduce rows (`ts_pieces`, `ts_pieces_min`, `ts_fuse`, `ts_duplex`).

## Summary

The three-rank exact two-shot can now run each block's words as a pipeline of pieces, send one piece and assemble
another in a single loop so that both directions of a link carry bytes at once, and take the duplex link weights,
under which the rank behind a link of half the rate owns no shard. Before, a rank sent, waited for its peers to
reduce, then received, one direction at a time. The defaults leave every launch as it was, and every setting gives
the same bits.

## Outcome

Untested (not promoted). VTP36, Qwen3.8-27B on XTX, XTX and R9700 (TP3, no peer access), four fused pieces with the
duplex weights and rad0033's `ts_once`, 2x ABBA against the layouts' pin: one-prompt time to first token -24% (w8a8)
and -27% (MXFP4) at 2048 tokens, 8 streams +14% and +17%; output bit-identical.

## Detail

See README.md.
