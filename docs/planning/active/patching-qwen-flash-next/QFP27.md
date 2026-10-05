---
id: QFP27
order: 0
plan: patching-qwen-flash-next
state: pending
created-at: '2026-10-05T02:13:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# Consolidate 1326/QFP24 async-input lifetime with upstream scheduler copy/event pipeline

## Description

QFP24 proposes a private pinned ring plus tracked Meta fanout to make 1326 safe for large asynchronous host->GPU inputs. Fresh upstream llama.cpp PR #29963 now implements a closely related lifetime primitive in the generic scheduler: user inputs use a single tensor copy per backend/copy slot, are uploaded before split submission, completion is recorded in `input_events[backend][copy]`, and the scheduler synchronizes the *next host copy* before the user can overwrite/reuse it. The same PR also makes transient host weights single-copy scheduler inputs and reuses their allocation after each split.

Do **not** land two independent lifetime systems. QFP27 is the consolidation/qualification item: determine whether #29963's existing scheduler copy generation is sufficient for 1326/QFP24, then reduce QFP24 to pinned allocation only if pinning still provides measured H2D/overlap benefit.

Ownership:
- QFP16/patch 1326: existing async host-input capability and measurements.
- QFP24: pinned-memory optimisation only after this qualification; no second generation/event protocol.
- QFP27: upstream #29963 compatibility, lifetime proof and consolidation decision.
- MET06/#29963: transient MoE host-weight policy; QFP27 must not duplicate expert-selection logic.

## Fresh upstream mechanism

PR #29963 (`pp-host-weights`, head `d4823011e342332459aeb7a5413b9448a6d3256d`, opened 2026-10-04) adds to `ggml_backend_sched`:

```cpp
ggml_backend_event_t input_events[GGML_SCHED_MAX_BACKENDS][GGML_SCHED_MAX_COPIES];
bool graph_in_flight;
int64_t n_moe_ids_max;
```

Before submitting splits it computes the next scheduler copy and waits only for that generation's previous input uploads:

```cpp
const int next_copy = (sched->cur_copy + 1) % sched->n_copies;
for (int b = 0; b < sched->n_backends; ++b) {
    if (sched->input_events[b][next_copy] != nullptr) {
        ggml_backend_event_synchronize(sched->input_events[b][next_copy]);
    }
}
```

It then uploads user inputs early and records completion on each destination backend stream:

```cpp
ggml_backend_tensor_set_async(
    sched->backends[split->backend_id], input_cpy,
    input->data, 0, ggml_nbytes(input));
...
ggml_backend_event_record(
    sched->input_events[b][sched->cur_copy], sched->backends[b]);
```

This is the same fundamental invariant QFP24 needs: source generation N cannot be reused until every destination DMA consuming generation N has completed. Prefer extending this invariant to Meta fanout rather than maintaining `bc_async_input_slot::done[]` as a second protocol.

## Steps

1. Rebase/port only the #29963 scheduler input-event portion onto the current BigCherry pin in an isolated experiment; do not import MoE placement policy for this test.
2. Instrument 1326/QFP24 source lifetime with `{cur_copy,next_copy,generation,backend,input_ptr,bytes,event_wait_us}` and prove which physical host allocation is associated with each scheduler copy.
3. Stress immediate source overwrite after submission for >=10,000 generations with 2/3/4 scheduler copies and 3 Meta children. Validate byte-generation tags on every child.
4. Determine whether Meta exposes one ordering stream/event that completes after all child `set_tensor_async` fanout. If yes, use one scheduler `input_event` per Meta backend/copy. If no, factor a Meta completion event/ticket internally, but attach it to the **existing scheduler generation**, not a QFP24 ring generation.
5. Compare pageable versus backend-host-buffer/pinned storage while holding scheduler copy depth and event protocol identical. Pinning is a transport optimisation, not lifetime correctness.
6. Measure copy-depth 2/3/4. Choose the smallest depth that hides >=95% of observed H2D latency without increasing peak VRAM/host-pinned memory unnecessarily.
7. On ring/copy reuse, retain slot-local synchronization only. Do not add a generic event-query API unless measured `input_event` reuse waits consume >=1% of prefill wall time.
8. If #29963 lifetime semantics pass and pinned memory adds <3% prefill/TTFT improvement, close QFP24 and fold only the upstream event-generation design into 1326.
9. If pinning adds >=3%, revise QFP24 implementation to allocate pinned backing for scheduler-owned copy slots while reusing #29963 events/generations; delete its proposed duplicate `bc_async_input_lane`, `bc_async_input_slot::done[]`, and tracked simple-backend protocol.
10. Re-run decode/MTP <=4 MiB to ensure no regression; this item must not widen the fast path without evidence.

## Detailed Solution & Technical Design

### One lifetime state machine

The desired state is scheduler-copy-owned:

```cpp
struct bc_input_generation {
    uint64_t generation;
    int copy_id;
    // backing storage is pageable or pinned; lifetime semantics are identical.
};

// Before producer/user may reuse copy_id:
wait(input_events[*][copy_id]);

// Populate host backing for copy_id, enqueue all destination uploads.
for (destination : destinations) {
    set_tensor_async(destination, dst_copy[copy_id], host_copy[copy_id]);
}

// Record after the final enqueue on each destination ordering domain.
record(input_events[destination][copy_id]);
```

Do not key lifetime by tensor pointer. Pointer identity can survive graph reuse while its contents belong to a new generation. `copy_id + generation` is the correctness identity; tensor/buffer identity is only allocation identity.

### Meta completion

The critical qualification question is whether `ggml_backend_event_record(meta_backend)` is ordered after all simple-child fanout. If current Meta event semantics do not provide that guarantee, add a private aggregate completion object inside Meta. It should be consumed by the scheduler's existing `input_events[b][copy]` seam, e.g.:

```cpp
bool ggml_backend_meta_record_fanout_completion(
    ggml_backend_t meta,
    ggml_backend_event_t aggregate,
    int copy_id);
```

Implementation may record child events and make `aggregate` synchronize/query all children, but QFP24 must not separately own those events. One source generation -> one scheduler completion handle.

### Pinned storage experiment

After correctness is proven, replace only backing allocation:

```cpp
auto buft = ggml_backend_dev_host_buffer_type(ggml_backend_get_device(dst));
auto buf  = buft ? ggml_backend_buft_alloc_buffer(buft, rounded_bytes) : nullptr;
```

Keep a bounded per-copy size-class pool. No per-ubatch `hipHostMalloc`, no pointer-keyed `thread_local` map. Measure `memcpy_us`, H2D enqueue-to-complete, overlap with compute, pinned high-water, and slot reuse wait.

### Why this is preferable to the current QFP24 draft

QFP24 currently proposes a second ring, its own generations, per-slot child events, a tracked Meta `set_tensor_async`, and potentially a new event-query proc. #29963 demonstrates that the generic scheduler already needs copy-indexed input completion for pipeline parallelism. Consolidating there removes two independent notions of when an input is safe to overwrite and makes ordinary user inputs and transient host weights share one tested lifetime invariant.

## Validation

Matrix:
- copy depth 2/3/4;
- pageable vs pinned backing;
- simple HIP backend vs 3-child Meta fanout;
- 1K/24K/80K/200K prefill, ub512/1024 where memory permits;
- MTP and no-MTP;
- two sequential requests plus concurrent server slots;
- deliberate delayed child stream and immediate producer overwrite.

Record: generation mismatches, wait_us per reused copy, H2D GB/s, copy-engine overlap, prefill t/s, TTFT, peak VRAM, pageable/pinned host high-water and fallback count.

Correctness gate: zero stale-generation reads in >=10,000 reuse cycles and no reuse before the slowest child completes.

Performance gate: adopt pinned backing only for >=3% prefill or TTFT improvement over the same #29963-style event lifecycle. Event-query work is allowed only if copy-reuse synchronization itself is >=1% of prefill wall.

## External References

- llama.cpp PR #29963: https://github.com/ggml-org/llama.cpp/pull/29963
- Current upstream release baseline on 2026-10-05: b11401 (`a7fb71f`). b11400 also explicitly moved mixed token/embedding batch handling to constant graph topology, reinforcing the project-wide rule that runtime data should not create unreserved graph variants.
- llama.cpp releases: https://github.com/ggml-org/llama.cpp/releases

## Related

QFP16, QFP24, MET06, QFP22. QFP27 should be closed once the lifetime protocol is consolidated into 1326/upstream-compatible scheduler semantics and QFP24 is either reduced to pinned backing or retired.

## Change Log

- 2026-10-05: Created after deep scan of QFP24 and fresh upstream #29963. Consolidates async-input lifetime onto scheduler copy/event generations before further pinned-ring work.
