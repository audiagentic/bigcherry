# rad0023_ar_timeline

## Summary

On a world without peer access (the host-staged transport), a three-rank decode all-reduce costs 0.4-0.6 ms a call,
far above its payload's PCIe time. `--profile-ops` gives a call's total only, so it cannot say whether the time is the
push of this rank's message into its peers' host slots, the wait for the peers' flags (their late arrival or the flag
round trip), or the read and fold of their messages. This patch records those phases.

With `R4D_AR_TIMELINE=DIR`, the wide one-shot (`r4d_ar_oneshot_wide_exact_kernel`, 3, 4 and 8 ranks) stamps the
device wall clock (100 MHz) four times in each of its first eight blocks:

| word | stamp |
|---|---|
| 0 | entry, after the sequence take |
| 1 | the push issued and drained (the handshake's first stamp, the existing `tl` hook) |
| 2 | every peer's flag seen (the handshake's second) |
| 3 | the fold stored |

A ring of 16384 entries per all-reduce instance lives in mapped pinned host memory. An entry is 36 int64 words: seq,
16-byte words, blocks, rank, then four stamps for each of eight blocks; -1 is unset. The kernel picks the entry from
the sequence number it already takes on the device, so recorded and replayed passes record too. At exit each ring is
written to `DIR/ar-tl-r<rank>-ws<world>-p<pid>-i<n>.bin`, n the ring's creation order across the whole process, so
it does not name the same all-reduce on every rank. `tools/lab/radiance/ar-timeline.py` matches a rank's rings by
their order on that rank, then entries by seq, and checks that the matched entries agree on size.

Unset or empty, the launcher passes a null ring: no stamp, no store, the same work and the same bits. Card clocks are
not aligned, so only differences within one card are meaningful; the smallest flag wait among a call's ranks bounds
the round trip, and the rest is a peer arriving later.

## Outcome

Untested (not promoted). A diagnostic for VTP31 experiment 2.

## Detail

- `libr4d/r4d_ar_oneshot_Nrank_exact.hip`: the kernel takes `tlring`; the four stamps; `artl_ring` (lazy, one ring per
  instance, keyed by its sequence counters) and `artl_dump` (atexit) in the host side.
- libr3 builds the same unit, so a mixed world's ranks all record.
- Source-pinned to radiance v1.17.0 (19d4e8fe); independent of the other patches.

### Promotion record

Model: Qwen3.8-27B MXFP4 (.rad), TP3 over two RX 7900 XTX and an R9700 (host-staged transport, no peer access)
Mechanics: tools/tests/patch/test_vtp31_ar_timeline.py passes and patch-lint is clean on the promotion PR; the lane builds carrying it build and serve.
Activation: with R4D_AR_TIMELINE=DIR the engine wrote per-rank timeline rings (ar-tl-r<rank>-ws3-p<pid>-i<n>.bin) and tools/lab/radiance/ar-timeline.py matched them by order and seq; VTP31.md experiment 2 (build 9e3cf8c7, logs tl-1011) split the one-shot into push, wait and fold (R9700 fold the long phase, about 120 us for 640 KiB; it arrives first and waits about 95 us for the XTX pushes).
Identity: diagnostic only; unset or empty, the launcher passes a null ring: no stamp, no store, the same work and the same bits; the TP3 greedy hashes and rad0024 results in VTP31.md were taken on builds that carry it.
A/B: no speed claim (off by default); the stamps are the result: per-call push/wait/fold totals in VTP31.md experiment 2 decided rad0024 (+5.0/+3.5/+6.1% at 4/6/8 streams).
