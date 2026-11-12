# rad0021_vtp14_weighted_twoshot_3rank

**Status:** untested
**Plan item:** VTP14

Scope: link-weighted shards for libr4d's exact three-rank two-shot all-reduce, as the engine already does at eight ranks.

## Summary

A rank behind a slow PCIe link owns a smaller shard of the two-shot, or none, and so moves fewer bytes over it. The
bits are the same for every table. Equal links keep the even split; decode-sized messages are unchanged.

## Outcome

Untested (not promoted). The expected effect is shorter prefill all-reduces on the x4 R9700 at TP3.

## Detail

See README.md.
