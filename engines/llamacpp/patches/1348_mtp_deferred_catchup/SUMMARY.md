# 1348_mtp_deferred_catchup

**Status:** validated  
**Plan item:** QFP31

## What it does

Prompt-only MTP catch-up is deferred by one target chunk. After target chunk k finishes, its unmasked NextN rows and token metadata are copied into one of two MTP-owned host snapshots. The draft decode is not run yet. The server submits target chunk k+1 first; then catch-up k runs on the same CPU thread while k+1 is executing on the target GPUs. The final pending catch-up is flushed before speculative begin/sampling.

`BIGCHERRY_MTP_DEFERRED_CATCHUP=0` restores the native synchronous MTP process path. First deferred use prints `BIGCHERRY_PATCH_HIT patch=1348_mtp_deferred_catchup`.

## Correctness boundaries

- Snapshot data is copied before the next target submit can overwrite llama.cpp's NextN output buffer.
- Draft catch-ups consume the same tokens/hidden rows in the same order as native MTP.
- Mixed/non-prompt and embedding batches flush pending work, then use native processing.
- 1322 look-ahead is generation-only in the composed path; any final prompt catch-up is flushed before the first ahead draft.
- `BIGCHERRY_MTP_DEFERRED_CATCHUP` and `BIGCHERRY_MTP_AHEAD` remain independent switches.
- Slot release/prompt clear/cache load invalidate pending work.
- Context shift and failed target decode drop the pending snapshot and poison affected sequences, disabling MTP drafting for the remainder of that request rather than using stale draft state.
- No worker thread, scheduler ring, graph-shape cache, or generic backend lifetime change is introduced.

## Qualification

Qualified on Brutus at b11474 via the QFP18 lightweight promotion tier: mechanics/lint, activation marker, one-binary
ABBA with complete separation at 8K/24K/98K, and greedy target identity. Full hardware record is in README.md and
`releases/evidence/qfp31-mtp-deferred-catchup.md`.

## Snapshot copy: timing and optional threads (2026-10-11, QFP42)

A kernel trace of Flash-Next prefill (run dp1) shows each target card with nothing queued for about 34 ms a
512-token chunk (81 gaps of 10 ms or longer, 2.8 s of a 29.6 s prefill at 24K). Kernels are queued ahead, so that
gap is host work done after the wait for chunk k returns and before chunk k+1's first kernels are submitted. The
synchronisation trace (run ps1) puts that wait in this patch's `llama_get_embeddings_nextn` call, as intended, and
the host profile (run pp2) shows the main thread in `memmove` for about 13 ms a chunk: the copy of the chunk's
hidden states into the deferred snapshot. The rest of the gap is the next graph (11.4 ms) and its inputs (4.3 ms),
which this patch does not own.

One switch, leaving the default path as it was: `BIGCHERRY_MTP_SNAPSHOT_THREADS=N` (1..16, default 1 = the plain
memcpy) splits the copy over N threads. The copy is already timed by 1346, which wraps these lines
(`BIGCHERRY_MTP_PROMPT_TIMING=1`, `snapshot_copy_ms`), so this patch adds no timing of its own; 1346's anchor
follows the new call.

Host threading conditions: N - 1 helper threads are created once with the MTP object and joined when it is
destroyed; none is created per chunk. Shared state is the copier's own job list, generation counter and running
count, all under its mutex. The helpers touch only the byte ranges they are handed, the ranges do not overlap, and
`copy()` returns only when every part is written, so neither buffer is used by a helper after the call. They make
no llama, ggml or HIP call. Not yet measured on hardware.
