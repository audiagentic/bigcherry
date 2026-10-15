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

1. Prove split-mode, copy count, Meta event support and exact current-pin source semantics before attempting any upstream transplant; #29963 is layer-pipeline-only, not a tensor-split solution.
2. Reuse existing 1325/1326 trace and admit further work only for >=1 ms AND >=3% exposed prefill input-handling cost.
3. Reuse QFP16/1326 as implementation owner and its existing pageable <=4 MiB control; never add a second scheduler generation or host-weight policy.
4. If qualified, implement a bounded pinned slot with **all-child** Meta completion or fail closed to pageable/synchronous copy before partial submission.
5. Run static admission and 10,000-generation source-shaped host tests before HIP work; test 2/3/4 slots, delayed children, cancel/reset and graph reuse.
6. Qualify on exact BigCherry GPU topology, multiple requests/ubatches, long context, MTP acceptance, logits/KLD and measured H2D/CPU timings.
7. Promote only >=5% median prefill/TTFT, positive CI95-low, <=2% decode regression, zero lifetime violations and <=64 MiB pinned; otherwise retire residual.
8. Recheck upstream PR #29963 only if it changes tensor-split admission and Meta event semantics; do not port unrelated MoE placement.

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

## 2026-10-13 exact-pin applicability correction (b11552; upstream #29963 head 273cd26)

**Source-confirmed disposition: upstream input-event pipeline is NOT an applicable lifetime fix for BigCherry's production `-sm tensor` / Meta input copies.** It remains an **open** upstream PR (updated 2026-10-12), not a merged pin or an AMD-qualified improvement.

- `src/llama-context.cpp::llama_context` enables `cparams.pipeline_parallel` only for **layer** split with multiple devices, sufficient GPU layers, KQV offload, no tensor overrides/cache and device async+event support. `sched_new(..., cparams.pipeline_parallel, ...)` passes that boolean into `ggml_backend_sched_new`.
- `ggml/src/ggml-backend.cpp::ggml_backend_sched_new` sets `n_copies = parallel ? GGML_SCHED_MAX_COPIES : 1`; PR #29963's `ggml_backend_sched_upload_user_inputs` and `input_events[backend][copy]` are used only when `n_copies > 1`. The production Meta tensor split has **one** scheduler copy; merely cherry-picking `input_events` does not activate it.
- `ggml/src/ggml-backend-meta.cpp::ggml_backend_meta_device_get_props` reports `caps.events=false`; `ggml_backend_meta_device_iface.event_new=nullptr`. `ggml_backend_meta_set_tensor_async` fans mirrored inputs to each child backend stream, without a Meta completion event. Even a forced multi-copy Meta experiment cannot interpret a null Meta event as completed H2D. A single child's event cannot protect source reuse while other children are reading it.
- BigCherry `1326_sched_async_host_inputs/patch.py` stages eligible contiguous non-weight inputs through scheduler-owned **pageable** `std::vector` memory and caps the shortcut at **4 MiB**. HIP's pageable-copy semantics explain why this path removes destination synchronization but does not prove genuine H2D overlap. Converting storage to pinned without child-completion tracking is unsafe. Existing v5c measurements remain historical (decode +9.4% at ~8K, +7.8% at ~64K; prefill -2.3%/-2% before the 4 MiB cap), not current-pin performance.

### Bounded owner implementation and decision gates

1. **No speculative port:** Do not transplant #29963's host-expert placement, `single_copy` aliases or pipeline-parallel allocation policy into tensor split. QFP16/1326 is the only implementation owner; QFP27 owns applicability; QFP24 owns *conditional* pinned backing, not another copy-generation scheduler. MET06 retains host-weight policy.
2. **Cheapest discriminator:** In existing 1325/1326 trace record `{phase, n_copies, split_backend, bytes, Meta-child-count, source-pinned, stage-pinned, memcpy_us, set_async_us, sync_us}` at ub512/1024/2048/4096. Confirm real prefill cost **>=1 ms AND >=3% of prefill wall** before any pinned-memory implementation. If either threshold fails, close residual without a hardware campaign.
3. **If qualified, extend 1326 in-place:** Use bounded per-scheduler, per-generation pinned staging (initial cap 64 MiB) only for large eligible inputs. Never reuse/overwrite slot G until **every** child backend consuming G has completed. Record child events after each destination enqueue; wait/join all child completions before slot reuse. If an event cannot be created/recorded, a child fails, or Meta split shape is unsupported, fall back to existing pageable/synchronous path **before submitting a partial copy**. A null event is not success. No global event-query API or second scheduler registry.
4. **Lifetime/graph safety:** Keep source bytes stable from CPU memcpy until the last child H2D read; generation must not be keyed solely by `ggml_tensor *` or graph pointer. Handle cancellation, reset, graph realloc/rebind, backend teardown, failed enqueue and slot wrap with wait-or-safe-fallback; do not free a live pinned buffer. Preserve compute ordering and per-rank bytes. No normal P2P exists on the dual-XTX/R9700/6900-XT topology.
5. **Host before GPU:** Static admission tests for `layer/tensor`, `n_copies=1/>1`, Meta `events=false`, 4 MiB threshold, null event and one-delayed-child; deterministic 10,000-generation 2/3/4-slot, 1/2/3-child host fixture with delayed completion, immediate producer overwrite, failure/cancel and allocation reuse. Then on idle hardware test two requests in one process, multi-ubatch prefill, long-context MTP/no-MTP, greedy/full-logit/KLD parity, all child completions, graph reuse and pinned high-water.
6. **Promotion/rejection:** Compare same-pin control 1326 with subject + pinned backing, four independent order-balanced sessions. Promote only with **>=5% median prefill/TTFT improvement**, positive CI95 lower bound, **<=2% decode regression**, zero stale generations/incorrect bytes and <=64 MiB pinned budget; otherwise retain safe <=4 MiB path and close QFP24/QFP27. Historical unsafe-source prefill gains are not evidence for promotion.

Upstream: [llama.cpp #29963](https://github.com/ggml-org/llama.cpp/pull/29963) (open, head `273cd26`, updated 2026-10-12). Other-engine mechanism: [vLLM multimodal H2D](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/inputs.py) pins dense CPU inputs before nonblocking transfer; it supplies neither Meta fanout completion nor transferable AMD speedup.

**Evidence performed:** 10/10 actual-source static assertions; 9/9 source-shaped host-model tests and Python compile. No repository pytest, patch application, HIP/Vulkan build, GPU run or new throughput measurement.

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

Performance gate: adopt pinned backing only for >=5% median prefill or TTFT improvement, positive CI95-low over the same #29963-style event lifecycle. Event-query work is allowed only if copy-reuse synchronization itself is >=1% of prefill wall.

## External References

- llama.cpp PR #29963: https://github.com/ggml-org/llama.cpp/pull/29963
- Current upstream release baseline on 2026-10-05: b11401 (`a7fb71f`). b11400 also explicitly moved mixed token/embedding batch handling to constant graph topology, reinforcing the project-wide rule that runtime data should not create unreserved graph variants.
- llama.cpp releases: https://github.com/ggml-org/llama.cpp/releases

## Related

QFP16, QFP24, MET06, QFP22. QFP27 should be closed once the lifetime protocol is consolidated into 1326/upstream-compatible scheduler semantics and QFP24 is either reduced to pinned backing or retired.

## Change Log

- 2026-10-05: Created after deep scan of QFP24 and fresh upstream #29963. Consolidates async-input lifetime onto scheduler copy/event generations before further pinned-ring work.
