---
id: QFP50
order: 50
plan: patching-qwen-flash-next
state: pending
created-at: '2026-10-10T16:51:24.805241+00:00'
breadth: ''
skill: advanced
created-by: claude
priority: P1
work: L
---

# Prefill pipeline, two batches deep: submit batch k+1 before waiting for batch k

## Description

In Flash-Next prefill each target card runs dry for about 34 ms a 512-token batch (kernel trace dp1: 81 gaps of 10 ms or longer, 2.8 s of a 29.6 s prefill at 24K; 99.7% of all gaps are about 3.5 us, so within a batch the queueing works). The batch-to-batch interval is 389 ms, so this is about 9% of prefill.

Why the cards run dry: the host only starts preparing batch k+1 after batch k has finished. The order today is: submit k (host 45.5 ms: graph build and allocation 11.4, inputs 4.3, submission 29.8) -> drafter catch-up for k-1 (30.3 ms, overlaps k) -> wait for k in llama_get_embeddings_nextn (280 ms; synchronisation trace ps1: 21.6 s of 23.9 s of all host waiting is this one site) -> snapshot copy (1.2 ms) -> back to the server loop -> build and allocate k+1, upload inputs, Meta rebuild (about 4 ms) -> first kernels of k+1. Everything after the wait is host work with the cards idle.

The host is idle for about 280 ms of every batch, so no extra thread is needed: the work has to be reordered so that batch k+1 is queued while batch k is still running.

Rejected on measurement: splitting the snapshot copy over threads (PR #133, closed): the copy is 1.18 ms, not the 13 ms first read off a profile without call chains; prefill unchanged. Rejected: moving the tensor split so the R9700 arrives earlier (TS 0.32,0.28,0.40 runs an XTX out of memory at ctx 245760).

## Steps

1. Inventory (above), written into this item.
2. Standalone probe first (tools/lab/hip-probes): a two-deep pipeline with reused buffer addresses, asynchronous inputs and an event-scoped wait, against the one-deep order, to confirm the gain and the ordering rules on these cards before touching the engine.
3. New patch (new mechanism, default off, e.g. BIGCHERRY_PREFILL_PIPELINE=1): A and B in ggml / the HIP backend, the nextn getter and decode in llama-context, C and D inside 1348 (its own package).
4. Lightweight tier: mechanics tests, marker, ABBA at 8K / 24K / 98K, greedy identity, at least twelve runs at 98K (the 1356 race only showed there), ar-prefill-stats.py and kernel-gap-stats.py before and after (the long gaps must go), 1346 prompt timing.

## Detailed Solution & Technical Design

New order, steady state, one batch always queued ahead:

1. build + allocate + submit batch k+1 while batch k is running;
2. wait for an EVENT that marks batch k's hidden-state copy to host as done (not a whole-stream synchronise, which would also wait for k+1);
3. snapshot k (1.2 ms), run the drafter catch-up for k (30 ms) while k+1 computes;
4. back to 1 for k+2.

Host work a batch (about 77 ms) is far below a batch's compute (about 280 ms), so the queue never empties.

What has to change:

A. `ggml_backend_sched_alloc_graph` synchronises the backends before it allocates (trace ps1: 76 calls from process_ubatch). Allocating batch k+1 while k runs is safe only because every write into the reused compute-buffer memory is stream-ordered behind k's kernels: kernels by construction, inputs through 1326's asynchronous input copies (BIGCHERRY_SCHED_ASYNC_INPUTS, on in the flashnext profile). Any host-side write that is not stream-ordered breaks this and must be found first (inventory below). The synchronise is skipped only for the pipelined prefill path.

B. A per-context event recorded on every device stream right after the batch's hidden-state `ggml_backend_tensor_get_async`; `llama_get_embeddings_nextn` waits on that event when the pipeline is on. Needs an event record / wait entry in the HIP backend and its Meta wrapper (one event a simple backend, wait for all).

C. 1348's `process_deferred` shifts by one batch: called after batch k+1 is submitted, it waits for k's event, snapshots k and runs catch-up k. The end-of-prompt flush then covers two outstanding batches instead of one.

D. The host buffer the hidden states land in: batch k+1's copy arrives about one batch time after k's, so the snapshot of k is taken long before; make it robust anyway with two host buffers by parity rather than relying on timing.

Inventory to do before code (each a read of the composed source, not an assumption): every synchronise reachable from `llama_context::decode` for a prompt batch; every host write into compute-buffer memory outside a stream (set_tensor paths without 1326, memset of outputs, Meta arena map, split copies between backends); what `ggml_backend_sched_reset` frees that a still-running batch's submission could reference (the Meta backend's per-graph state, RCCL group calls, HIP graph instances); KV cache writes of k and reads of k+1 (stream-ordered on the same card; the cross-card sum is enqueued on the compute stream today).

## Code Samples & Guidance



## Files



## Validation

Greedy text identical to production at 8K / 24K / 98K (the mechanism is a reordering; any difference is a bug). kernel-gap-stats.py: gaps of 10 ms or longer drop from about one a batch to near zero. Prefill gain with complete separation in ABBA. Decode and draft acceptance unchanged. Cancel, context shift, cache reuse and a prompt shorter than two batches each exercised (the outstanding-batch bookkeeping).

## Effort & Risk



## Standards



## Acceptance Criteria

- The inventory is recorded and every non-stream-ordered write is either absent or handled.
- The standalone probe result is recorded.
- The patch is promoted (lightweight tier) or rejected with its measurement; identity holds in every run including twelve at 98K.
- The before / after gap table is in the patch summary.

## Notes

Expected ceiling about 9% of prefill time (34 of 389 ms). Independent of the cross-card sum work (QFP49, where staggering half-batches recovered 95% of the exchange cost in the standalone probe) and it composes with it: a staggered pair of half-batches is itself a pipeline within a batch. No new host thread, so it is not exposed to the 1356 class of race; the 1356 bisect (needs HIP graphs and the fusion pass) is still worth reading before step 3 because this also changes when graphs are captured relative to a running batch.

## Change Log

- 2026-10-10T16:51:24.805241+00:00 (created-by): Created by claude
