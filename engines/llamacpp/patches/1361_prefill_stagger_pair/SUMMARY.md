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

## b-stg11 real-graph follow-up

Code-level root cause of the first-call `different Meta subgraph shapes` decline: the two 256-token ubatches are
not output-symmetric. `bc_prepare` computes `n_outputs` independently. For a normal prompt A usually requests no
logits while B, the last ubatch of the decode call, carries the requested last-token logits/output head. The model
builder therefore gives B an output tail A does not have; depending on where its split reduction lands this changes
the final Meta subgraph or adds a subgraph. The old 1361 gate required identical total subgraph counts and rejected
the pair before any staggered work.

1361 now compares only the per-card node count and reduction-boundary op for the common Meta prefix, pairs only the
common reduction-producing subgraphs, then runs the remaining A tail followed by the remaining B tail. A mismatch
logs `subgraphs A/B`, first differing subgraph index, node count/op for both sides, and the number of reductions that
can still pair. Zero comparable reductions still fail closed. All reductions, including asymmetric tail reductions,
are provider-preflighted before model work begins.

Pair-owned pinned input staging no longer has 1326's 4 MiB policy cap and no longer assumes a fixed byte count.
Each input records current bytes and capacity. Capacity only grows; before replacing either pinned slot, 1361 waits
that slot's previous H2D-completion events, then reallocates both slots. A staging decline names half, input index,
tensor name, bytes now, bytes before, and reason.

`BIGCHERRY_PREFILL_STAGGER_TIMING=1` emits one
`BIGCHERRY_PREFILL_STAGGER_HOST` line per successful pair with host microseconds in prefix splits, pinned staging
(including capacity checks/grows), Meta capture/rebuild, static-view re-homing, interleaved/tail submit, total, plus
staged bytes and sum count. With the diagnostic off no timers are sampled. The per-call shape check was reduced from
scanning every node op to the reduction-boundary properties needed by the pairing contract.

The Meta harness now deliberately gives B one extra split/reduction output tail while A has none. It also gives each
half a CPU-produced Meta input and rebuilds that input larger on every pair through 6 MiB, comparing every pair
bitwise to sequential, then restores the small input before the timing arm. These are source-level tests only until
the harness is rebuilt on the target hardware.

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

## Scheduler host-prefix qualification (PR #147 follow-up)

The normal Flash-Next prompt graph has a host-side input split followed by a final Meta split. 1361 now admits matching A/B scheduler shapes with exactly one Meta split, last. All preceding A splits and then B splits use the unchanged upstream scheduler split loop (input-copy ordering, host-weight callbacks, async compute, events and cross-backend waits); final Meta input copies precede the staggered Meta compute. Host-prefix work starts only after Meta has captured both graphs and preflighted every reduction/provider, so unsupported routes perform no prefix work. The B host split consumes B's separately materialized token/embedding/position inputs, not the output of A's Meta split or A's KV/recurrent state. Existing 1359 pipelining likewise permits next-batch host preparation before prior-batch Meta compute finishes. A-before-B dependencies are enforced during final Meta computation on the shared per-card streams.

On first structural/provider decline the context latches `bc_1361_declined` and subsequent batches take the unchanged sequential `process_ubatch`/graph-reuse route; one already-prepared pair is executed sequentially. Declines are logged once per distinct reason, including reduction index for RCCL route failure. A successfully engaged pair still rebuilds/allocates two graph results each candidate call; hardware measurement must quantify this overhead. No additional threads, feature remains default-off.
