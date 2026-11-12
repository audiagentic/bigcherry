# rad0024_ts3_decode

**Status:** untested
**Plan item:** VTP31

Scope: the exact three-rank one-shot all-reduce on a host mailbox with settled link weights (rad0021).

## Summary

From the all-reduce row's `ts_min` elements, as upstream's four-rank one-shot does, the three-rank one-shot runs the
link-weighted exact two-shot at any size, so the rank behind the slowest link reads about one message instead of two.
Same bits; `--tune-pin all_reduce:ts_min=0` turns it off.

## Outcome

Untested (not promoted). Built for VTP31 experiment 3.

## Detail

See README.md.
