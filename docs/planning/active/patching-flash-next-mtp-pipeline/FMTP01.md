---
id: FMTP01
order: 0
plan: patching-flash-next-mtp-pipeline
state: pending
created-at: '2026-10-04T00:48:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P0
work: M
---

# Instrument and fail-closed gate the MTP overlap lane

## Description

Prove that the production Flash-Next lane has a real asynchronous target-submit window and a physically independent MTP execution lane before changing scheduling.

Do not add a worker or asynchronous state machine here.

## Steps

1. Add disabled-by-default `common_params_speculative_draft::mtp_ahead = 0`, CLI `--spec-mtp-ahead N`, env `LLAMA_ARG_SPEC_MTP_AHEAD`; bridge is additional to `N`.
2. Add a read-only MTP capability query reporting `is_mem_shared`, `chain_heads`, `n_mtp_layers`, probabilistic mode and effective front depth.
3. v1 eligibility fails closed unless: draft-mtp, ahead requested, non-shared, single-head, non-probabilistic, one active sequence.
4. For the production qualification lane additionally require target/draft device sets to be disjoint and record whether any model/meta buffers are shared across contexts. `!is_mem_shared` alone does not prove physical isolation.
5. Split timing of target verification into:
   - target `llama_process()` submit/enqueue host time;
   - target `llama_synchronize()` wait time;
   - total target device interval from profiler.
6. Record front MTP draft, draft rollback/truncate, `common_speculative_process()` replay/reseed and target acceptance times.
7. Prove with rocprof that after `llama_process(ctx_tgt)` returns, target kernels remain in flight until explicit synchronization. c061 uses `ggml_backend_sched_graph_compute_async`; this is the basis for the worker-free FMTP03 design.
8. Record target and draft backend/device names and any draft-context work appearing on target GPUs.
9. Add bounded activation/rejection reasons: disabled, not_mtp, shared_memory, chain_heads, probabilistic, multi_seq, device_overlap, shared_buffer.
10. Keep detailed timing in trace/evidence; add server metrics only for bounded aggregates.

## Detailed Solution & Technical Design

c061 server currently executes target process and synchronization together:

```cpp
queue_tasks.yield_to_queue([&]() {
    ret = llama_process(ctx_tgt, LLAMA_PROCESS_TYPE_DECODE, batch.view.get());
    if (ret == 0 && has_output) {
        llama_synchronize(ctx_tgt);
    }
});
```

The underlying context submits with `ggml_backend_sched_graph_compute_async()`. FMTP01 must measure the process-return-to-sync interval separately; do not insert any extra synchronization for telemetry.

Suggested capability shape:

```cpp
struct common_speculative_mtp_caps {
    bool valid = false;
    bool memory_shared = false;
    bool chain_heads = false;
    bool probabilistic = false;
    int32_t n_mtp_layers = 0;
    int32_t n_max = 0;
};
```

Device-disjointness can remain a server/qualification check if common speculative code cannot inspect both context device sets cleanly.

Branch constraints:
- 1268 owns front-depth adaptation.
- 1280 is rejected; corrected sidecar metadata is required.
- 1293 showed sync-count reduction alone is not a performance objective.

## Files

- `common/common.h`
- `common/arg.cpp`
- `common/speculative.h/.cpp`
- `tools/server/server-context.cpp`
- optional bounded metrics files
- patch package + mechanics tests
- `config/experiment-contracts.toml`

## Validation

Offline:
- argument bounds/defaults;
- capability-query matrix;
- ahead=0 creates no new scheduling path;
- fail-closed device-overlap/shared-buffer checks where observable;
- patch apply/idempotence/missing-anchor tests.

Hardware:
- target tensor split + separate MTP GPU;
- shallow/deep context;
- rocprof timeline proving target process returns before target GPU completes and MTP device is disjoint.

## Acceptance Criteria

- Ahead defaults off with unchanged behavior.
- Production lane is explicitly eligible/rejected.
- `target_submit_us` and `target_sync_wait_us` are separately measurable without new syncs.
- Profiler proves an exploitable target-in-flight window and where MTP work executes.
- Evidence is sufficient to choose worker-free FMTP03 or reject it before implementation.
