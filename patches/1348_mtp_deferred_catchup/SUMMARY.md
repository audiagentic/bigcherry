# 1348_mtp_deferred_catchup

**Status:** untested  
**Plan item:** QFP31

## What it does

Prompt-only MTP catch-up is deferred by one target chunk. After target chunk k finishes, its unmasked NextN rows and token metadata are copied into one of two MTP-owned host snapshots. The draft decode is not run yet. The server submits target chunk k+1 first; then catch-up k runs on the same CPU thread while k+1 is executing on the target GPUs. The final pending catch-up is flushed before speculative begin/sampling.

`BIGCHERRY_MTP_DEFERRED_CATCHUP=0` restores the native synchronous MTP process path. First deferred use prints `BIGCHERRY_PATCH_HIT patch=1348_mtp_deferred_catchup`.

## Correctness boundaries

- Snapshot data is copied before the next target submit can overwrite llama.cpp's NextN output buffer.
- Draft catch-ups consume the same tokens/hidden rows in the same order as native MTP.
- Mixed/non-prompt and embedding batches flush pending work, then use native processing.
- Slot release/prompt clear/cache load invalidate pending work.
- Context shift and failed target decode drop the pending snapshot and poison affected sequences, disabling MTP drafting for the remainder of that request rather than using stale draft state.
- No worker thread, scheduler ring, graph-shape cache, or generic backend lifetime change is introduced.

## Qualification

Expected primary A/B: production Flash-Next prompt throughput with MTP, default-on versus `BIGCHERRY_MTP_DEFERRED_CATCHUP=0`. Generated text must be unchanged. Hardware/compile qualification is pending.
