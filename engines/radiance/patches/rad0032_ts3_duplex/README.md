# rad0032_ts3_duplex

**Status:** untested
**Plan item:** VTP36

Scope: the exact two-shot all-reduce of a three-rank world whose links differ (host-staged or peer access), under two
new `--tune-pin` axes of the all-reduce rows.

## Summary

The three-rank exact two-shot can now run each block's words as a pipeline of pieces (`ts_pieces`), so both
directions of a link carry bytes at once, and can take the duplex link weights (`ts_duplex`), under which the rank
behind a link of half the rate owns no shard. Both exist upstream for eight ranks only. The defaults (one piece, the
settled weights) leave every launch as it was, and every setting gives the same bits.

## Outcome

Untested (not promoted). See README.md for the measurements.

## Detail

See README.md.
