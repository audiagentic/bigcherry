# rad0024_ts3_decode

## Summary

VTP31 experiment 2 (rad0023) split the three-rank host-mailbox decode all-reduce into phases. On two RX 7900 XTX at
x16 and an R9700 at Gen4 x4, at eight streams (a 320 KiB message), the R9700 spends 120 us of its 218 us call reading
its two peers' messages over its link; the flag round trip is 2 us. The one-shot is what decode always takes at
three ranks: the two-shot row starts above `R4D_AR_WIDE_ONESHOT_MAX_ELEMS` (524288 elements), and the link weights
rad0021 settled apply only from `R4D_AR8_MIN_ELEMS_DEFAULT` (262144).

Upstream's four-rank exact one-shot already hands a message to the exact two-shot from the row's `ts_min` axis
(default 32768 elements). This patch gives the three-rank one-shot the same rule where it pays, a host-mailbox instance
whose weights are settled: from `ts_min` elements it runs the exact three-rank two-shot with the settled table at any
size. The slow-link rank owns the smallest shard, so it reads about 0.9 of a message instead of two, for one more
handshake.

## Detail

| case | path |
|---|---|
| host mailbox, weights settled, numel >= `ts_min` | weighted exact two-shot |
| below `ts_min`, `ts_min` 0, peer access, or no settled weights | one-shot, unchanged |
| the shard pair does not fit the slot, or the shape is refused | one-shot, unchanged |

The two-shot folds every word in fp32 in ascending rank order like the one-shot, so the output bits do not change.
As in upstream's four-rank handover, the two-shot runs on the one-shot's own instance, with its flags, sequence
counters and slots: launches on an instance are ordered on its stream, and a slot that holds the whole message holds
the largest weighted shard pair. The handover adds no state. The default `ts_min` (32768 elements) takes decode from
two streams at a 5120 hidden size (8 rows); one stream (4 rows) stays on the one-shot.

`--tune-pin all_reduce:ts_min=0` restores the one-shot everywhere for an A/B; any other legal `ts_min` moves the
threshold.

## Files

- `libr4d/r4d_ar_entry.hip`: the handover in `r4d_rad_ar_oneshot_3rank_exact`; the three-rank two-shot entry becomes
  a body (`ts3_exact`) that takes the settled weights at any size when the handover calls it.

Source-pinned to radiance v1.17.0 (19d4e8fe), after rad0021.

## Outcome

Measured in VTP31 experiment 3 (MXFP4 TP3, 2x ABBA against `ts_min=0`): +5.0% / +3.5% / +6.1% at 4/6/8 streams, one
stream and TTFT unchanged, output bit-identical. State stays untested until promoted.
