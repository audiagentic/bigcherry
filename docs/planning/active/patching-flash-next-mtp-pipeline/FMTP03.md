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

# Overlap MTP continuation with target verification

## Description

Run FMTP02's bounded `bridge + tail` continuation on the separate MTP GPU while the target verifies the current front draft. Preserve a single owner for `ctx_dft` and join before any existing path reuses, restores or destroys that context.

## Steps

1. Add one persistent worker per speculative context; no per-token or per-round thread creation.
2. Publish immutable jobs containing only copied seed state, `seq_id`, epoch and requested tail depth.
3. After normal front drafting, if an eligible seed exists, launch/publish the continuation **before** target `llama_process()`.
4. Delay the current post-draft draft-context rollback/truncation until the worker has completed. The worker needs the speculative frontier created by the front.
5. After target verification has synchronized, join/collect the worker **before** `common_speculative_process()` touches `ctx_dft`.
6. Then run the existing rollback/truncate and authoritative `common_speculative_process()` path unchanged.
7. Join before any operation that can mutate/restore/free `ctx_dft`: begin/reset, state restore, prompt checkpoint restore, sleep/model unload, destruction and error recovery.
8. Add per-sequence epoch fencing. Reset/replay/context-shift marks jobs stale; a late completion is copied out only when epoch/frontier still match.
9. v1 cancellation is logical, not GPU preemption. A stale in-flight continuation may finish but is discarded. Maximum work is bounded by `mtp_ahead`.
10. Record worker queue delay, ahead device time, overlap window and join overhang.

## Detailed Solution & Technical Design

Use a persistent thread with a one-job mailbox. `std::jthread` is preferred when the project/toolchain supports it; otherwise use `std::thread` with explicit shutdown. Avoid `std::async` because execution policy/thread creation is implementation-dependent.

```cpp
struct mtp_ahead_job {
    uint64_t epoch;
    llama_seq_id seq_id;
    int32_t n_tail;
    common_mtp_ahead_seed seed;
};

struct mtp_ahead_mailbox {
    std::mutex mu;
    std::condition_variable cv;
    std::optional<mtp_ahead_job> pending;
    std::optional<common_mtp_ahead_result> complete;
    bool stop = false;
    bool running = false;
};
```

No worker pointer may refer to `server_slot`; slots are server-thread-owned. Jobs contain copied scalar/vector state only.

### Ownership timeline

```text
server thread                         MTP worker
-------------                         ----------
common_speculative_draft(front A)
publish(seed) ----------------------> resume bridge + tail
build/submit target verify A          ctx_dft exclusively owned here
llama_synchronize(ctx_tgt)
join/collect <------------------------ publish copied token result
rollback/truncate ctx_dft
common_speculative_process(A)
post_decode accept/reject A
```

The server may interact with unrelated queue/HTTP state during target decode, but no code path may enter `ctx_dft` while the job is running.

### Where to launch

Current server `pre_decode()` drafts before adding the sampled+draft tokens to the target verification batch. The launch should happen after front draft/checkpoint metadata are fixed but before the target decode begins. It must not run before the front's required speculative context state exists.

### Where to join

Current `decode()` calls target `llama_process()`, synchronizes when outputs are needed, and then calls `common_speculative_process(spec, batch.view)`. Join after target synchronization and before that call. This is the largest safe window without duplicating the draft context.

### Rollback ordering

Today server code may restore/truncate `ctx_dft` immediately after drafting because no more draft work is expected before target verify. FMTP03 must move that action after join. Do not delete it: continuation state is speculative and must still be discarded before authoritative processing.

### Error paths

Worker failure result:

```cpp
struct common_mtp_ahead_result {
    ...
    bool ok = false;
    int32_t decode_rc = 0;
};
```

An ahead failure must not fail the request if the original synchronous speculative path remains healthy. Record failure, invalidate result, rollback draft context and continue normally.

## Code Samples & Guidance

RAII join guard around draft-context entry points is preferable to scattered assertions:

```cpp
void common_speculative_join_ahead(common_speculative * spec) {
    if (!spec) return;
    spec->mtp_ahead.join_if_running();
}
```

In debug builds, assert worker is idle immediately before `common_speculative_process()` enters MTP `process()`.

## Files

- `common/speculative.h`
- `common/speculative.cpp`
- `tools/server/server-context.cpp`
- patch package + worker mechanics tests

## Validation

Offline concurrency tests with a fake resume callback:
- publish/join repeated thousands of times;
- reset while work is running => old epoch result rejected;
- destruction joins cleanly;
- failure result falls back;
- mailbox never exceeds one pending job in v1;
- no stale server-slot pointers.

Run ThreadSanitizer on the fake worker test where supported.

Hardware:
- rocprof timeline must visibly overlap target verification kernels on target GPUs with continuation kernels on the MTP GPU;
- trace must show no overlapping `ctx_dft` process/replay call from server thread;
- compare ahead off/on greedy output before FMTP04 promotion is enabled (ahead is computed then discarded in this item).

## Effort & Risk

L / high. Main risk is draft-context lifetime/ownership, not the worker primitives.

## Acceptance Criteria

- Non-zero real target/MTP overlap is demonstrated.
- One persistent worker; no hot-path thread creation.
- Existing rollback/replay still executes after join.
- `common_speculative_process()` never races the worker.
- Sleep, unload, reset, error and destruction paths join safely.
- Ahead computation can be enabled-and-discarded with exact control output.
