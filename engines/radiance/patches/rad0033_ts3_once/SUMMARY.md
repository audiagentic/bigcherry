# rad0033_ts3_once

**Status:** untested
**Plan item:** VTP36

Scope: the exact two-shot all-reduce of a three-rank host-staged world (no peer access), under the new `--tune-pin`
axis `ts_once` of the all-reduce rows.

## Summary

In the two-shot's second hop a rank now writes its reduced shard once, into its own host scratch, and each peer
assembles it from there; before, it wrote one copy per peer. The default (0) leaves every launch as it was, and the
setting gives the same bits. Over peer access the axis is ignored.

## Outcome

Untested (not promoted). VTP36: with the duplex weights of rad0032 the XTX ranks' 32-row decode call falls from 81 to 65 us;
the prefill gain is carried by rad0032. See README.md.

## Detail

See README.md.
