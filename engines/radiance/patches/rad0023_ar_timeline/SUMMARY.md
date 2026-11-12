# rad0023_ar_timeline

**Status:** untested
**Plan item:** VTP31

Scope: a diagnostic timeline for libr4d's exact one-shot all-reduce at three, four and eight ranks, on only while
`R4D_AR_TIMELINE=DIR` is set.

## Summary

Each launch stamps, in its first eight blocks, entry, push drained, every peer's flag seen and fold done, into a
pinned host ring indexed by the launch's device sequence number; each ring is written under DIR at exit. Unset, the
kernel takes no stamp and does the same work.

## Outcome

Untested (not promoted). Built to split a host-staged decode all-reduce into push, peer wait and fold (VTP31
experiment 2).

## Detail

See README.md.
