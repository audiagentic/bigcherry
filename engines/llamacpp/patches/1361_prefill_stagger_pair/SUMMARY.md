# 1361_prefill_stagger_pair

**Status:** untested
**Plan item:** QFP49

## What it does

`BIGCHERRY_PREFILL_STAGGER=<half tokens>` (0/unset = off) enables one narrow prompt path: a decode call that the
memory context has split into exactly two ubatches A and B, each exactly `half` tokens, may execute the tensor-split
Meta backend one subgraph out of phase. The server must still be launched with `-b 2*half -ub half`; this patch does
not change batch defaults.

For each Meta subgraph i the schedule is:

`end(A[i-1]); compute A[i]; begin(A[i]); end(B[i-1]); compute B[i]; begin(B[i])`.

A[i] is always enqueued before B[i] on every simple backend compute stream. That order is a correctness requirement,
not a performance preference: B may read KV rows, recurrent/GDN/SSM state, or other model state written by A at the
same layer.

The pair path is preflighted before either graph starts. It requires identical Meta subgraph counts and an RCCL
begin/end route for every reduction in both graphs. Internal-pipeline, cpu-root, adaptive-host, and generic Meta
fallback reductions reject the pair before execution; the caller then uses the existing sequential graph order.

No host thread is added.

## Two graph/allocation sets

A continues to use the existing llama scheduler, graph result, and 1340 physical Meta arena. With the flag enabled,
the context creates a second scheduler and graph result sized for `half`, reserves it independently, and logs the
reserved B compute-buffer size per backend. B's Meta physical compute arena is a second 1340 arena owner; A and B
therefore cannot alias the same per-device compute allocation while both graphs are live.

The two llama schedulers own distinct logical Meta compute buffers with independent `stc_compute[2]`
generations, and the Meta pair entry retains independent A/B per-card subgraph metadata until completion.
Static Meta buffers (KV/recurrent state) are intentionally shared: their *external views* do not have independent
generations. Capturing A can rotate the shared static-buffer container after B's graph has been allocated.
Before capturing B, the pair re-homes B's static external-view chains into the current container without clearing
A's captured mappings; B's rebuild then rotates the inactive container. The view headroom is 32 with the flag on
(two graph sets between evaluations), retaining upstream 16 with the flag off. The physical compute arenas remain
separate 1340 owners for A and B. Repeated-pair/graph-replay pointer lifetime is still a hardware qualification risk.

## AllReduce begin/end

1361 exposes 1360's RCCL transfer split through backend-registry proc addresses. Meta owns two independent transfer
state objects (A and B); their lifetime spans begin to end, including 1360's pooled BF16 temporaries. The existing two
1360 event slots permit one A and one B reduction to be outstanding.

Provider admission is decided before the pair starts. A graph that would choose any non-RCCL provider or Meta's
generic fallback is not partially staggered.

## 1359 interaction

1359's secondary pinned NextN copy is extended only while a 1361 pair is being extracted: A writes its rows at the
pair output offset, B writes the following rows, and the 1359 completion event is recorded after B's copy. With
`BIGCHERRY_PREFILL_STAGGER=0`, 1359's existing single-ubatch `offset == 0` gate and bytes are unchanged.

## Identity reference

A staggered pair must be bit-identical in logits/text to the same build run sequentially with
`-b 2*half -ub half` and `BIGCHERRY_PREFILL_STAGGER=0`. It is intentionally not compared with `-ub 2*half`,
which is a different graph/batching reference.

## Shared-state inventory

- KV cache: A and B write/read it through normal model compute streams. On every card A[i] is enqueued before B[i];
  the per-card compute stream supplies the layer-local order, while each prior cross-card sum is ended before that
  half advances to its next dependent subgraph.
- Recurrent/GDN/SSM state: same ordering rule as KV. A's layer-i work precedes B's layer-i work on every card; neither
  half advances past a required sum until 1360 end has placed the transfer-completion wait on the compute stream.
- 1359 pinned NextN buffers: output-copy backend stream writes A's range then B's range; the fence events are recorded
  only after B's copy, so a fenced reader sees both halves. Slot reuse still waits on the prior slot event.
- 1360 pool allocations: each outstanding reduction state owns its own per-device BF16 pool allocation until end
  enqueues post-conversion; A/B transfer states are independent and use different 1360 event slots.
- A compute buffers: owned by the primary scheduler and bound to 1340 arena slot A; only A graph tensors use them.
- B compute buffers: owned by the secondary scheduler and bound to 1340 arena slot B; only B graph tensors use them.
  This is the deliberate memory cost of staggering (measured target configurations are about 459 MiB/card at half
  256 and about 1,021 MiB/card at half 512).
- RCCL tensors/events: compute producer -> 1360 input event -> transfer-stream RCCL -> output event -> compute
  consumer. A and B each carry their own caller-owned begin/end state.
- Host bookkeeping: decode, scheduler pair dispatch, Meta pair dispatch, and 1360 slot bookkeeping all run on the
  existing caller thread; no worker/helper thread is introduced.

## Activation and fallback

The marker `BIGCHERRY_PATCH_HIT patch=1361_prefill_stagger_pair` is emitted once, only when the first pair actually
starts. One ubatch, odd/tail batches, decode-sized batches, callbacks, non-causal work, embeddings/rerank, unequal
subgraph shapes, and unsupported communication providers remain on the existing sequential path.
