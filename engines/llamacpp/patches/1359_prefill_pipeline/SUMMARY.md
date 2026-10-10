# 1359_prefill_pipeline

**Status:** untested
**Plan item:** QFP50

## What it does

With `BIGCHERRY_PREFILL_PIPELINE=1` and an MTP drafter, the server submits prompt batch k+1 before the drafter hook
collects batch k's hidden states. `llama_context::decode` copies an unmasked batch's nextn rows a second time into
one of two pinned host buffers, by decode-call parity, and records a ggml backend event behind that copy on every
device of the tensor-split backend. 1348's hook, entered after batch k is submitted, collects batch k-1 through
`llama_get_embeddings_nextn_fenced` (which waits on that batch's events only), runs its catch-up while k computes,
and leaves k outstanding. Off by default; unset, every path is as before.

## Why

Kernels are queued ahead of the cards, so a card only runs dry where the host could not submit yet. In Flash-Next
prefill that is once a batch: the hook waits for batch k in `llama_get_embeddings_nextn` (synchronisation trace ps1:
21.6 s of the 23.9 s the host waits in a 24K prefill), and only then does the server build, allocate and submit
batch k+1. The kernel trace (dp1) shows 81 gaps of 10 ms or longer on each target card, 2.8 s of a 29.6 s prefill.
The same prefill with no drafter is 14.6% faster (run nomtp1: 1,530 / 1,534 against 1,340 / 1,334 t/s), which is
the upper bound for this patch: it removes the wait's position in the order, not the drafter's own outputs.

Nothing else in the batch path waits for the previous batch (`ggml_backend_sched_alloc_graph` synchronises only when
an allocation has to move: 76 calls, 114 ms in all), and the host is otherwise idle for about 280 ms of a 389 ms
batch, so no thread is added.

## Ordering and shared state

- One host thread, as before. The order of work on each card's stream is unchanged: batch k+1's kernels, inputs and
  copies are enqueued behind batch k's.
- The fenced copy lands in a buffer no later batch writes until the second decode call after it; before a buffer
  is reused, decode waits for its events, which also keeps the pipeline two batches deep and no deeper.
- The buffers are pinned host memory from the output device's host buffer type, as llama's own output buffer is.
- The outstanding batch is collected by `flush_deferred` (so by the prompt boundary flush, the non-deferred path and
  the hook itself) and dropped by `reset_deferred` under the rule 1348 uses for a pending snapshot.
- A backend without events, or a masked nextn output, gives no fence: the hook then waits as 1348 does.
- 1346's prompt timing counts the synchronous path only; with the pipeline on its chunk counters stay at zero.

Standalone check of the mechanism on these cards (`tools/lab/hip-probes`, test `pipeline`, run hp6): batch k+1
queued into the same card buffer while k runs, the host waiting on an event behind k: 15.5 ms a batch against 18.5
on an RX 7900 XTX (the batch computes in 15.4 ms, the host preparation is 3 ms), every result right.

## Still to show

Offline mechanics and patch-lint; on hardware the marker `BIGCHERRY_PATCH_HIT patch=1359_prefill_pipeline`, greedy
text identical to the flag off at 8K, 24K and 98K (a reordering: any difference is a bug), at least twelve runs at
98K, an ABBA on one binary, `kernel-gap-stats.py` before and after (the long gaps must go), decode and acceptance
unchanged, and cancel / context shift / a prompt shorter than two batches exercised.
