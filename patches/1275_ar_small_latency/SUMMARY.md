# 1275 — small AllReduce latency controls

**Status:** untested
**Plan item:** `PGC10`

Adds opt-in latency controls for single-chunk mapped-host AllReduce while keeping unset/default behavior equivalent to pristine b11233: `BIGCHERRY_AR_SLOT_SYNC=host|stream|none` (default `host`), `BIGCHERRY_AR_SMALL_BLOCKS={1,2,4,8}` (default `8`), and `BIGCHERRY_AR_SMALL_THREADS={128,256}` (default `256`).

`none` removes the pool-wrap host event waits only for single-chunk small collectives. `stream` instead enqueues waits on each old-generation `ev.ker` before that slot can be re-recorded. Multi-chunk reductions retain host synchronization. The arrival allocation/stride remains sized for `GGML_CUDA_AR_KERNEL_BLOCKS=8`; geometry controls only reduce the launched grid.

When `BIGCHERRY_PATCH_TRACE` is set, the patch emits a once-per-process activation marker plus per-small-AR `host_enqueue_us` and `slot_wait_us` CPU timing. Timing adds no device or host synchronization. The conditional root3 edits apply only when 1244 was materialized first.

No hardware validation is claimed.
