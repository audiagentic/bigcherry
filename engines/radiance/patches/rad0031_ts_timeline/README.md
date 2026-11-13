# rad0031_ts_timeline

## Summary

On a three-rank world without peer access, every prefill all-reduce and every multi-stream decode all-reduce is the
link-weighted exact two-shot (rad0021, rad0024). `--profile-ops` gives a call's total only, and rad0023's timeline
covers the one-shot. This patch records the two-shot's phases.

The two-shot kernel already has upstream's per-launch timeline (`tl_all`, used by the eight-rank selftest): word 0
after the sequence take, an (arrive, release) pair for each handshake, and a last word after the final assemble.
With `R4D_AR_TIMELINE=DIR` set, every launch of the kernel that was not given that buffer stamps the same words into
a ring instead:

| word of a block record | content |
|---|---|
| 0 | the block's handshake sequence number |
| 1 | the message, in 16-byte words |
| 2 | blocks, pieces << 8, rank << 16, world << 24 |
| 3 | this rank's shard, in 16-byte words |
| 4 | start, after the sequence take |
| 5, 6 | first push issued and drained; every peer's first flag seen |
| 7, 8 | reduce and second push drained; every peer's second flag seen (one pair more for each further piece) |
| 9 | assemble stored (after the last pair) |

A ring of 8192 entries per all-reduce instance lives in mapped pinned host memory. An entry is four block records
(the first four blocks) of 4 + R4D_AR_TL_WORDS int64 words; -1 is unset. The kernel picks the entry from the sequence
number it already takes on the device, so recorded and replayed passes record too. At exit each ring is written to
`DIR/ar-ts-r<rank>-ws<world>-p<pid>-i<n>.bin`. `tools/lab/radiance/ar-ts-timeline.py` reads them.

Unset or empty, the launcher passes a null ring: no stamp, no store, the same work and the same bits. Card clocks are
not aligned, so only differences within one card are meaningful.

## Outcome

Untested (not promoted). A diagnostic for VTP36.

## Detail

- `libr4d/r4d_ar_twoshot_Nrank_exact.hip`: the kernel takes `tlring` behind upstream's `tl_all`; `tstl_ring` (lazy,
  one ring per instance, keyed by its sequence counters) and `tstl_dump` (atexit) on the host side.
- libr3 builds the same unit, so a mixed world's ranks all record.
- Source-pinned to radiance v1.17.0 (19d4e8fe); after rad0021.
