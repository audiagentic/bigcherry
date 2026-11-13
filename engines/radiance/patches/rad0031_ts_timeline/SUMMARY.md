# rad0031_ts_timeline

**Status:** untested
**Plan item:** VTP36

Scope: a diagnostic timeline for libr4d's exact two-shot all-reduce at three, four and eight ranks, on only while
`R4D_AR_TIMELINE=DIR` is set.

## Summary

Each launch of the exact two-shot stamps, in its first four blocks, its start, each handshake's arrival and release,
and its end, into a pinned host ring indexed by the launch's device sequence number; each ring is written under DIR at
exit. Unset, the kernel takes no stamp and does the same work. rad0023 does the same for the one-shot, which no longer
carries a three-rank world's prefill or multi-stream decode (rad0021, rad0024).

## Outcome

Untested (not promoted). Built to split a host-staged three-rank two-shot into first push, peer wait, reduce with
second push, second wait and assemble (VTP36).

## Detail

See README.md.
