---
id: FMTP03
order: 0
plan: patching-flash-next-mtp-pipeline
state: pending
created-at: '2026-10-04T00:48:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P0
work: L
---

# Overlap target verification and MTP work on one host thread

## Description

Exploit c061's asynchronous target graph submission directly. Split target `llama_process()` from `llama_synchronize()`, execute bounded MTP work between them, then resume the existing authoritative rollback/reseed path.

Do **not** add a persistent worker in v1 unless FMTP01 hardware evidence disproves this execution model.

## Steps

1. Preserve the existing target verification batch construction and speculative checkpoint logic.
2. Submit target verification with `llama_process(ctx_tgt, ...)` but do not immediately synchronize.
3. If the current front was freshly drafted, call FMTP02 `continue_live_front()` while target GPU work is in flight.
4. If the current front was promoted from a prior ahead result, call FMTP02 `replay_known_front_and_continue()` while target GPU work is in flight.
5. Call `llama_synchronize(ctx_tgt)` after bounded MTP work completes.
6. Then perform the existing draft rollback/truncate and `common_speculative_process()` authoritative replay/reseed unchanged.
7. On any MTP ahead failure, discard the ahead result and continue the target request normally.
8. Keep all work on the server processing thread in v1; no mailbox, background thread, cross-thread context use or worker lifetime state.
9. Record target submit time, MTP overlap work time, target sync wait and derived overhang.
10. Preserve queue responsiveness by using the repository's existing `yield_to_queue` mechanism around the new submit/work/sync sequence without allowing another task to mutate the same slot/context between its phases.

## Detailed Solution & Technical Design

Current c061 server shape:

```cpp
queue_tasks.yield_to_queue([&]() {
    ret = llama_process(ctx_tgt, LLAMA_PROCESS_TYPE_DECODE, batch.view.get());
    if (ret == 0 && has_output) {
        llama_synchronize(ctx_tgt);
    }
});
```

The context itself submits with `ggml_backend_sched_graph_compute_async()`. Proposed shape:

```cpp
queue_tasks.yield_to_queue([&]() {
    ret = llama_process(ctx_tgt, LLAMA_PROCESS_TYPE_DECODE, batch.view.get());
    if (ret != 0) {
        return;
    }

    if (eligible_ahead) {
        ahead_ok = promoted_front
            ? common_speculative_replay_front_and_ahead(spec.get(), ...)
            : common_speculative_continue_live_ahead(spec.get(), ...);
    }

    if (has_output) {
        llama_synchronize(ctx_tgt);
    }
});
```

Exact placement can differ if `yield_to_queue` requires smaller scopes, but three ordering invariants are mandatory:

1. target GPU submit happens before MTP overlap work;
2. target synchronization happens after MTP overlap work;
3. no authoritative `ctx_dft` rollback/process happens until the MTP work is finished.

Ideal critical-path time after target submit is:

```text
T_overlap = max(T_target_gpu, T_mtp_work)
T_overhang = max(0, T_mtp_work - T_target_gpu)
```

This is the same device-level overlap a two-thread design seeks, without thread-safety risk.

### Fresh-front round

The live lease captured in FMTP02 remains valid because no `ctx_dft` mutation occurs between front draft and continuation.

### Promoted-front round

There is no live speculative lease from the previous round; it was intentionally destroyed by authoritative rollback. FMTP02 replays the known front from current authoritative MTP seed and continues beyond it during this overlap window.

### Failure path

If MTP work fails, still synchronize target, then run existing rollback/reseed and acceptance. Ahead failure must be a performance miss, not a request failure.

### Fallback worker criterion

Only revisit a worker if profiler evidence shows the production target backend completes/synchronizes inside `llama_process()` despite c061's async graph API, or if server queue semantics make the same-thread split impossible without unacceptable responsiveness regressions. Any worker design then needs a separate thread-safety proof.

## Code Samples & Guidance



## Files

- `common/speculative.h/.cpp`
- `tools/server/server-context.cpp`
- patch package + scheduling tests

## Validation

Offline:
- fake target submit/sync test proves MTP callback runs strictly between them;
- fresh and promoted paths select the correct FMTP02 primitive;
- ahead failure still reaches target sync + ordinary speculative processing;
- no MTP work runs when ahead=0.

Hardware:
- rocprof visibly shows target kernels on target GPUs overlapping MTP kernels on the draft GPU;
- target submit host call returns before target GPU interval completes;
- target sync wait shrinks by approximately the hidden MTP duration until overhang begins;
- no cross-thread llama/backend calls exist in v1.

## Effort & Risk



## Standards



## Acceptance Criteria

- Real cross-device overlap without a worker thread.
- Existing authoritative rollback/reseed remains after target sync.
- Ahead failure is fail-open to ordinary target correctness.
- Target/MTP overlap timing matches the `max(T_target, T_mtp)` model within profiler noise.
- No ahead=0 behavior or performance regression outside measurement noise.

## Notes

Prep 2026-10-04 - code anchors (vendor/llama.cpp/tools/server/server-context.cpp): target decode ~L3681 queue_tasks.yield_to_queue([&]{ ret = llama_process(ctx_tgt, ...); if (ret == 0 && has_output) llama_synchronize(ctx_tgt); }) - the ahead call goes between process and synchronize inside the same yield; 1317 (spec round timing) already splits submit/sync timers here - compose with it. Speculative reseed after the decode: ~L3747 common_speculative_process(spec, batch.view). Gate 0 numbers: target sync wait 25.5 ms (10K) / 31.4 ms (80K) per round vs ahead work = continuation 4 steps (bridge + 3) ~2.1-2.9 ms/step on the 6900 = ~8.5-11.6 ms (fresh) or forced replay 3 + continuation 4 = ~15-20 ms (promoted) - both fit inside the sync window. The 6900 is otherwise idle during target verify. Patch number: 1322_mtp_ahead_overlap. Async proof (FMTP01 step 7) still to capture with rocprof in the first 1322 run.

## Change Log

- 2026-10-04T06:32:25.863452+00:00 (updated-by): Updated: section:notes
