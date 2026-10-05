---
id: FMTP01
order: 0
plan: patching-flash-next-mtp-pipeline
state: completed
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

## Code Samples & Guidance



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

## Effort & Risk



## Standards



## Acceptance Criteria

- Ahead defaults off with unchanged behavior.
- Production lane is explicitly eligible/rejected.
- `target_submit_us` and `target_sync_wait_us` are separately measurable without new syncs.
- Profiler proves an exploitable target-in-flight window and where MTP work executes.
- Evidence is sufficient to choose worker-free FMTP03 or reject it before implementation.

## Notes

2026-10-04 Gate 0 (RV4216, owner-approved; absorbs QFP08's decode-overlap scope and its synctrace evidence: ~12 syncs per draft call, context-wide synchronize per step in llama_get_embeddings_nextn_ith). One instrumented binary, identical 10K/80K corpora, two modes.
BASELINE n_max=3 per round: target_process enter/return; target first/last kernel; target_sync enter/return; common_speculative_process subspans (hidden read/copy, draft catch-up, rollback/KV, accept, sampler); serial fresh-draft us; target logits D2H bytes (n_output_rows*n_vocab*4) and us; target sampler us; committed tokens; rocprof timeline of target + draft kernels + DMA/AR.
CALIBRATION n_max=7 (acceptance only, never timing): P(A1..A3), P(A4|A1..A3) (bridge), E[accepted A5..A7 | A1..A4] (promoted-front yield), per-position draft time.
Derive p_hit, Y_fresh, Y_prom, D_serial, C_reseed, C_logits, W_async (target_sync_done - target_process_return, with kernels in flight), W_hidden (target token final - authoritative hidden ready).
Decides, in order: F reseed/sync elimination (build if C_reseed >= 0.5 ms; 1-4%), C device argmax for target verify (if C_logits >= 0.5 ms/round; 0.5-3%; purpose-built, upstream backend sampling unsupported under tensor split), B early authoritative-hidden draft catch-up (if W_hidden useful; 2-8%), D 1268 EV signal (0-2%), FMTP02-05 only if bootstrap lower bound >= 2% throughput (committed tokens / wall time). FMTP06/unconditional FMTP not before that. QFP08 keeps draft-prefill overlap (TTFT).

2026-10-04 Gate 0 timing (1317, flashnext-gate0-d10k/d80k, b-det = v3 + 1312 + 1313, greedy, n_max 3; timer overhead nil: 41.1 vs 41.2/40.6 ms/step). Per-round medians ~10K / ~80K: round 38.4 / 47.9 ms; draft 6.36 / 8.67 ms (16 / 18%); target submit (llama_process host time) 5.06 / 5.83 ms, mean 6.7 / 7.1 (17 / 15%); target sync wait 25.5 / 31.4 ms (63%); common_speculative_process 0.75 / 1.28 ms (2 / 3%); target sample-and-accept 0.67 / 0.87 ms (1.5%). Acceptance 59 / 69%, full-front rounds 36 / 58%, tokens/round 2.76 / 3.07.
Decisions: F (reseed/sync) <= ~2-3% ceiling, C (device greedy) <= ~1.5% -> both low priority. B (early hidden catch-up) small: catch-up is ~1 ms; the cost is the fresh draft itself. FMTP gross ceiling = P(full)*P(bridge)*draft: ~0.25*6.4 = 1.6 ms (~4%) at 10K, ~0.41*8.7 = 3.5 ms (~7%) at 80K before promoted-front yield loss; workload-dependent -> keep FMTP02-05 gated on the n_max 7 calibration (not yet run). NEW top lever: target submit host time 5-7 ms/round (15-17%) - graph build/scheduling per verify step; owned by new item QFP16.

2026-10-04 owner direction: pursue all Gate 0 options (small wins count) - F, C, B, FMTP calibration and QFP16. Order by effort: (1) FMTP n_max 7 calibration run (queue-fmtp-calib.sh; summary prints P(prefix>=k), P(full front), P(bridge|full), promoted-tail yield); (2) F + draft-loop sync reduction (QFP08 synctrace: ~12 syncs per draft call, context-wide synchronize in llama_get_embeddings_nextn_ith) - this also attacks the 6.4-8.7 ms serial draft itself, not only the 0.75-1.3 ms reseed; (3) C device argmax for target verify; (4) QFP16 target submit host time after the perf profile; (5) B; (6) FMTP02-05 per calibration.

2026-10-04 Gate 0 calibration (flashnext-calib7b-d10k/d80k; depth-7 arm at CTX 131072 because the 8-token verify OOMs the R9700 at 240K; same prompt corpus as the n_max 3 timing runs): ~10K / ~80K: P(accepted prefix >= k) k1..k7 = .69 .58 .44 .37 .32 .29 .23 / .82 .63 .53 .42 .35 .30 .26; P(full 3-front) 0.44 / 0.53; P(bridge | full) 0.85 / 0.80; p_hit 0.37 / 0.42; promoted-tail yield E[accepted of 3 | bridge] 2.26 (75%) / 2.17 (72%) vs fresh n_max 3 acceptance 59% / 69% - the promoted front is NOT worse than a fresh front (conditioning on full-front + bridge selects predictable stretches and outweighs the depth decay). Projected FMTP saving = p_hit x serial draft: 0.37 x 6.4 = 2.4 ms of a 38.4 ms round (~6%) at 10K; 0.42 x 8.7 = 3.7 ms of 47.9 ms (~7.6%) at 80K, before overhang/contention, which the 25-31 ms target sync window should hide. Caveat: the calibration continued the live chain; FMTP02 forced replay may give a slightly different hidden trajectory. Decision: gate met on the point estimate (single prompt); proceed with FMTP02-05, keeping FMTP05's online EV control and the FMTP07 ABBA as the real acceptance gate.

2026-10-05 close-out: acceptance met. Ahead is default off (BIGCHERRY_MTP_AHEAD, patch 1322); submit and sync time are separately measured (1317: target submit 5.06 / 5.83 ms, sync wait 25.5 / 31.4 ms at ~10K / ~80K) without new syncs; the exploitable window is shown (63% of a round is target sync wait with the 6900 XT idle) and confirmed by the first 1322 screen (ahead host ~16-21 ms per round hidden under target verify); the n_max 7 calibration met the proceed gate (p_hit 0.37 / 0.42). The decision this item existed to make - worker-free FMTP03 - was taken and implemented as 1321 + 1322. Remaining work is tracked in FMTP03/FMTP04/FMTP05/FMTP07.

## Reviews

- RV4216

## Change Log

- 2026-10-04T03:26:42.716009+00:00 (updated-by): Updated: section:notes
- 2026-10-04T05:13:34.635207+00:00 (updated-by): Updated: section:notes
- 2026-10-04T05:24:47.114772+00:00 (updated-by): Updated: section:notes
- 2026-10-04T06:29:30.213802+00:00 (updated-by): Updated: section:notes
- 2026-10-05T10:04:08.471912+00:00 (updated-by): Updated: section:notes
- 2026-10-05T10:04:22.266126+00:00 (state-transition): State: pending → completed
