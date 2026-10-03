# 1275_ar_small_latency

**Status:** evaluated
**Plan item:** PGC10

Small mapped-host AllReduce latency controls over pristine b11233. Defaults
preserve pristine behavior: `BIGCHERRY_AR_SLOT_SYNC=host`,
`BIGCHERRY_AR_SMALL_BLOCKS=8`, and `BIGCHERRY_AR_SMALL_THREADS=256`.
`slot_sync=none` bypasses the pool-wrap host event waits only for a
single-chunk mapped-host reduction; copy-engine and multi-chunk paths retain
the host waits. Geometry accepts blocks `1|2|4|8` and threads `128|256` while
leaving the fixed `GGML_CUDA_AR_KERNEL_BLOCKS=8` arrival-ring layout unchanged.

Validation required: apply, HIP build on gfx1100, activation marker,
correctness, and paired performance A/B. Stream slot sync, timing traces, and
1244/root3 composition are intentionally deferred.
