---
id: QFP24
order: 0
plan: patching-qwen-flash-next
state: pending
created-at: '2026-10-05T00:00:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# Safe pinned scheduler input ring for asynchronous large prefill host->GPU copies

## Description

Patch 1326/QFP16 proved that synchronous split-input handling is expensive and that asynchronous host->device input copies can materially improve serving. The first zero-copy/pinned-source version also showed prefill gains (+3.7% / +7.2% in the recorded v5 screen), but it was unsafe because callers can rewrite graph-input host storage before true asynchronous DMA consumes it. The safe fix stages through scheduler-owned pageable memory; that preserved decode gains but made large prefill inputs slower, so 1326 now caps the fast path at 4 MiB. Its own result identifies the missing mechanism: a scheduler-owned pinned staging ring whose slots are not reused until every destination backend has consumed them.

QFP24 owns that prefill extension. QFP16 remains the target-verify submit diagnosis/acceptance item and 1326 remains the implementation base; do not add another scheduler input-copy path. The new mechanism should replace 1326's per-input `thread_local unordered_map<tensor*, vector<uint8_t>>` staging only for large inputs where pinned DMA + overlap beats the current synchronous path.

## Steps

1. Extend 1326 timing with per-input `{name, bytes, source backend, destination split, mirrored fanout count, staging memcpy us, enqueue us, completion/next-use distance}` on prefill ubatches. Rank the large inputs; current expectation is KQ/QSA masks around ~10 MiB/ubatch, but use measured names/sizes.
2. Allocate scheduler-owned pinned staging slots with stable addresses. Start with 3 slots per size class or per logical input-copy lane. The caller copies source bytes into a free slot synchronously on CPU, then the split backend asynchronously fans that pinned slot to its device copies.
3. Record completion on every destination that reads the slot. For Meta MIRRORED destinations, one slot cannot be reused until all participating GPU streams have passed the H2D copy. Use backend/device events or an explicit Meta copy-completion event array; no `backend_synchronize()` in the steady-state fast path.
4. Add generation numbers to prevent ABA reuse. Slot selection is `(logical_lane, generation % depth)`; if the next slot is still live, fall back to the existing 1326 pageable/synchronous path rather than host-waiting indefinitely.
5. Size policy: preallocate bounded capacity from the graph's observed/max eligible input sizes or a small set of powers-of-two. Do not `hipHostMalloc` per ubatch. Cap total pinned memory and expose telemetry; oversized inputs fall back safely.
6. Update Meta `set_tensor_async` fanout so completion ownership is observable. Prefer one event per destination stream recorded immediately after the copy. Do not add device-wide events or a global synchronize.
7. Sweep ring depth `{2,3,4}`, pinned cap and input-size threshold. Compare against upstream synchronous, current 1326 pageable staging and pinned ring. The ring is a prefill optimization only if end-to-end wall improves after including the CPU memcpy into pinned storage.
8. Test sequential ubatches, prompt-cache reuse, two concurrent server slots if supported, and source buffers that are themselves pinned. Slot lifetime must depend only on scheduler-owned storage, never caller lifetime.
9. If prefill wins, fold the ring into patch 1326 rather than creating a parallel patch. Update QFP16 notes with the final result; QFP24 can close once 1326 owns the qualified implementation.

## Detailed Solution & Technical Design

### Ownership and slot lifecycle

The current 1326 fast path copies the input to `std::vector<uint8_t>` then calls `ggml_backend_tensor_set_async()`. On HIP, pageable host memory makes the call effectively consume/copy enough data before return to make caller reuse safe, but that destroys the intended prefill overlap. Replace large-input staging with persistent pinned storage:

```cpp
struct bc_async_input_slot {
    void * host_ptr;              // hipHostMalloc / backend host-pinned buffer
    size_t capacity;
    uint64_t generation;
    bool active;
    backend_event done[MAX_META_RANKS];
};

struct bc_async_input_lane {
    bc_async_input_slot slots[4];
    uint32_t next;
};
```

Use the project's backend abstraction for host-pinned allocation and events where available; do not leak HIP types into generic scheduler interfaces unnecessarily.

Lifecycle for generation `g`:

```text
CPU producer complete
      |
memcpy source -> pinned slot[g % depth]
      |
set_tensor_async(meta/split, slot.ptr)
      |-- H2D rank0 -- record done0
      |-- H2D rank1 -- record done1
      `-- H2D rank2 -- record done2

slot becomes reusable only after all required done events have completed
```

The host `memcpy` is intentionally synchronous because it establishes independence from caller storage. It should be much cheaper than waiting for three H2D copies and allows the caller to proceed immediately after staging.

### Nonblocking slot reuse

Do not call `hipEventSynchronize` for every slot. Before choosing a slot, query its completion events. If all complete, reuse. If not, try another slot. If the ring is full, either use 1326's existing safe fallback or only then wait if measurement proves bounded waiting is better. Default to fallback for the first version to avoid turning a throughput optimization into host serialization.

Use monotonic generations for diagnostics and safety. A stale event from generation `g-depth` must never authorize reuse for generation `g` unless that slot's exact completion records correspond to the previous owner.

### Meta fanout

For a MIRRORED split input, `ggml_backend_meta_set_tensor_async` fans the same host pointer to each child backend. Add an internal completion-return/record hook rather than assuming enqueue means consumed. If the existing backend event API can record on each child stream immediately after `set_tensor_async`, use it. Otherwise add a private Meta helper used by the scheduler fast path:

```cpp
bool ggml_backend_meta_set_tensor_async_tracked(
    ggml_backend_t meta,
    ggml_tensor * dst,
    const void * pinned_src,
    size_t size,
    ggml_backend_event_t * completion_events,
    size_t * n_events);
```

This is illustrative. Prefer extending an existing event/copy helper over a public API if possible.

### Allocation policy

Pinned memory is a finite system resource. Start with a global cap such as 256 MiB and measured size classes. For example, a 3-slot ring for one 10 MiB mask is ~30 MiB, acceptable; duplicating that for every tensor identity is not. Key lanes by destination input-copy buffer/shape class, not source tensor pointer, so graph rebuilds do not leak a new ring.

Record `pinned_reserved`, `pinned_active`, slot waits/fallbacks, bytes staged and average CPU memcpy bandwidth. A ring that rarely reuses the same lane or falls back >10% should not be promoted without redesign.

## Code Samples & Guidance

Primary implementation must remain in 1326 anchors:

- `ggml/src/ggml-backend.cpp`: `ggml_backend_sched_compute_splits`, replacing/extending the current `bc_staging` block.
- `ggml/src/ggml-backend-meta.cpp`: tracked async fanout/completion for mirrored and supported split states.
- backend host-buffer/event helpers in existing GGML backend APIs; use them before direct `hipHostMalloc` if they expose portable pinned memory.

Pseudo fast path:

```cpp
if (bc_async_inputs && eligible_large_host_input(input, input_cpy)) {
    ggml_backend_synchronize(input_backend); // CPU producer must be done
    auto * slot = ring.try_acquire(ggml_nbytes(input));
    if (slot != nullptr) {
        memcpy(slot->ptr, input->data, nbytes);
        if (set_tensor_async_tracked(split_backend, input_cpy, slot->ptr, nbytes, slot->done)) {
            slot->active = true;
            continue;
        }
        ring.release(slot);
    }
}
// existing 1326/upstream fallback
```

The `input_backend` synchronize is retained only where the source is produced asynchronously; plain user/CPU-written inputs may already be ready. Optimize that separately only with evidence.

## Files

`patches/1326_sched_async_host_inputs/patch.py` and its tests/evidence; `ggml/src/ggml-backend.cpp`; `ggml/src/ggml-backend-meta.cpp`; existing backend event/host-buffer implementation; QFP16 for submit-path evidence; QFP17 for prefill ABBA integration.

## Validation

Unit/mechanics: slot depth 2 with deliberately delayed destination completion; source buffer overwritten immediately after staging; verify destination bytes retain the staged generation. Exercise mirrored 3-rank fanout, one-rank split, ring-full fallback, tail size, resize and graph rebuild.

Hardware: profile 10K/80K/200K fills at outer ubatch 512/1024 where available. Compare synchronous baseline, 1326 pageable staging and pinned ring. Record scheduler input wall, CPU memcpy wall, H2D overlap, prefill t/s, TTFT, PCIe copy-engine utilization and pinned-memory high-water.

Decode/MTP control: retain 1326's validated <=4 MiB path unchanged or prove the ring does not regress it. Greedy/reference output must be identical because only transport/lifetime changes.

## Effort & Risk

M. The code is localized, but lifetime bugs can cause silent corruption. Main risks are slot reuse before the slowest Meta child copy finishes, excessive pinned-memory reservation, event-query overhead and host memcpy becoming the new bottleneck.

## Standards

Scheduler-owned lifetime; bounded pinned memory; stable addresses; no caller-lifetime dependency; no per-ubatch pinned allocation; no global/device synchronization in steady state; explicit safe fallback; implementation folded into 1326 after qualification.

## Acceptance Criteria

- Immediate overwrite/reuse stress shows no stale/corrupt destination data across >=10,000 staged generations.
- Large prefill split-input handling drops >=30% or representative prefill improves >=3% versus current safe 1326, with no decode regression >1%.
- No steady-state backend/device synchronize is added for destination completion; slot ownership is event/generation based.
- Pinned allocation is bounded/configured and leak-free across graph rebuilds/repeated requests.
- Qualified code is consolidated into 1326; no second scheduler async-input implementation remains.

## Notes

This is the concrete follow-up already called out by the 1326 hardware result. It is separated from QFP16 because QFP16's primary objective is decode/MTP target-submit host time, while QFP24 is specifically the large-input prefill lifetime/overlap extension of the same mechanism.

## Change Log

- 2026-10-05T00:00:00+00:00 (created-by): Created by agent from QFP16/1326 prefill follow-up.
