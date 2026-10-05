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

Exploit asynchronous target graph submission directly: submit target verification, run bounded MTP ahead work on the dedicated draft GPU while target GPUs execute, then synchronize target and resume the authoritative rollback/reseed path.

After patch 1321, FMTP03 must use the existing single-head `draft()` forced-front/live-tail primitive rather than introduce the older `continue_live_front()` / `replay_known_front_and_continue()` helper API or a durable live-front lease. This consolidates scheduling around one MTP loop and removes duplicate speculative-state machinery.

Do **not** add a persistent worker in v1 unless hardware evidence disproves same-thread device overlap.

## Steps

1. Preserve target verification batch construction and speculative checkpoint semantics.
2. Submit target verification with `llama_process(ctx_tgt, ...)`; do not immediately synchronize.
3. During the target GPU interval, invoke the 1321 single-head MTP draft path with a private ahead result, `forced = current_front`, and bounded `n_tail`.
4. Forced-front replay reconstructs the MTP frontier from the current authoritative seed; the tail is generated in the same draft call and remains disjoint from the target verify set.
5. Synchronize target after bounded ahead work.
6. Perform existing authoritative rollback/truncate, acceptance and `common_speculative_process()` unchanged.
7. Ahead failure discards only the ahead result and continues the request normally.
8. Keep all calls on the server processing thread; no mailbox, worker, cross-thread context use or lease lifetime state.
9. Record target submit time, forced replay time, tail time, target sync wait, and overhang.
10. Use existing `yield_to_queue` semantics without permitting another task to mutate the same slot/context between submit, ahead work and sync.

## Detailed Solution & Technical Design

### Consolidation after 1321

The previous FMTP03 design referenced two proposed FMTP02 APIs: `continue_live_front()` for a fresh front and `replay_known_front_and_continue()` for a promoted front. Patch 1321 instead landed the lower-risk primitive directly in the existing single-head draft loop:

- `dp.forced`: feed known front tokens through the MTP head in order;
- `dp.n_tail` / `dp.tail`: continue beyond `n_max` in the same `draft()` call;
- forced tokens bypass `p_min`;
- chained/shared-KV modes reject the new fields;
- default-null/zero fields preserve existing behaviour.

FMTP03 should therefore **not** add the obsolete helper pair, a second MTP loop, or `mtp_live_lease/frontier_gen` state. The scheduling unit is one ordinary MTP draft invocation configured for forced replay + tail. This makes fresh and promoted fronts use the same correctness model: reconstruct from the authoritative seed, then speculate ahead.

This costs replay work for a fresh front versus a hypothetical live continuation, but current calibration leaves enough target slack to test the simpler design first. The measured target sync window is ~25.5 ms at 10K and ~31.4 ms at 80K; forced replay of a depth-3 front plus four continuation steps was estimated at ~15-20 ms on the 6900. If profiling confirms that work is hidden, durable live-front continuation has no Amdahl justification and should be removed from FMTP02 rather than implemented.

### Scheduling shape

```cpp
queue_tasks.yield_to_queue([&]() {
    ret = llama_process(ctx_tgt, LLAMA_PROCESS_TYPE_DECODE, batch.view.get());
    if (ret != 0) {
        return;
    }

    if (eligible_ahead) {
        llama_tokens ahead_front;
        llama_tokens ahead_tail;
        common_speculative_draft_params adp = make_private_ahead_params(...);
        adp.forced = &current_front;
        adp.n_tail = ahead_depth;
        adp.tail   = &ahead_tail;
        ahead_ok = common_speculative_draft_ahead(spec.get(), adp, ahead_front);
        // ahead_front is reconstruction-only; only ahead_tail is retained for promotion.
    }

    if (has_output) {
        llama_synchronize(ctx_tgt);
    }
});
```

The exact wrapper name may differ. Prefer a narrow adapter that invokes the existing 1321 draft implementation; do not clone its transition loop.

Mandatory ordering:

1. target graph submit precedes MTP replay/tail work;
2. target synchronization follows bounded MTP work;
3. no authoritative draft rollback/reseed occurs until ahead work is complete;
4. only the tail is persisted as an ahead proposal; replayed `forced` tokens never become duplicate target output.

Critical path:

```text
T_overlap  = max(T_target_gpu, T_forced_replay + T_tail)
T_overhang = max(0, T_forced_replay + T_tail - T_target_gpu)
```

### Adaptive depth: exploit slack, do not over-draft

Static `n_tail=4` is only the first lane. Add timing-only policy after correctness is proven:

```text
budget_ms = max(0, EWMA(target_gpu_ms) - safety_ms)
step_ms   = EWMA(mtp_step_ms)
replay_ms = front_len * step_ms
n_tail    = clamp(floor((budget_ms - replay_ms) / step_ms), 0, configured_max)
```

Use prior-round EWMAs only; never synchronize target early to obtain a current-round timing estimate. Start with a 2 ms safety margin. This converts otherwise idle target slack into useful proposals while bounding overhang when context/model timing changes. Policy belongs in FMTP03 scheduling, not in generic speculative code.

### Failure and cancellation

Ahead generation is opportunistic. Decode failure, `p_min` stop, cancellation, insufficient budget or unsupported draft mode yields no promoted tail. Target verification still synchronizes and the normal authoritative path proceeds. Never turn an ahead miss into a request failure.

## Code Samples & Guidance



## Files

- `tools/server/server-context.cpp`: submit/ahead/sync ordering, timing and bounded policy.
- `common/speculative.h/.cpp`: only a narrow adapter if 1321 cannot be invoked safely from server scheduling; no second MTP loop.
- `patches/1322_mtp_ahead_overlap/`: scheduling patch and mechanics tests.
- FMTP02: follow-up cleanup of obsolete live-lease/helper design after hardware proof.

## Validation

Offline:
- fake target submit/sync proves ahead callback occurs strictly between them;
- ahead uses 1321 forced-front + tail path for both fresh and promoted fronts;
- replayed front is not duplicated into persisted ahead tail or target output;
- ahead failure/cancel still reaches target sync and ordinary speculative processing;
- `ahead=0` is byte/behaviour equivalent to baseline;
- adaptive depth never exceeds configured max and returns zero when replay cost consumes budget.

Hardware (gfx1100/1201 target + 6900 draft):
- rocprof shows target kernels and MTP kernels overlapping in wall time;
- target submit returns before target GPU interval completes;
- sync wait shrinks by approximately hidden MTP duration;
- measure fresh/promoted separately at 10K/80K/160K;
- report `target_gpu_ms`, `replay_ms`, `tail_ms`, `sync_wait_ms`, `overhang_ms`, tail promotion rate and accepted promoted tokens/token;
- sweep static tail 1/2/4/6, then adaptive policy;
- no cross-thread llama/backend calls.

## Effort & Risk



## Standards



## Acceptance Criteria

- Real cross-device overlap with no worker thread.
- Existing authoritative rollback/reseed remains after target sync.
- No duplicate MTP transition implementation or live-front lease machinery.
- Ahead failure is fail-open to ordinary target correctness.
- >=70% of forced-replay + tail wall time is hidden under target verification on representative 10K and 80K lanes, or end-to-end decode improves >=5% with <=1 ms median overhang.
- Adaptive policy must not regress any representative lane >2% versus best fixed-depth baseline.
- Ahead disabled has no measurable performance regression and identical greedy output.

## Notes

2026-10-04 prep: server target decode is inside `queue_tasks.yield_to_queue`; 1317 already separates submit/sync timing and should be composed rather than duplicated. `common_speculative_process()` authoritative reseed remains after target verification.

2026-10-05 consolidation review: branch head `da30ae8` landed 1321 with forced promoted-front replay + live tail inside the existing single-head MTP draft loop. This makes FMTP03's older two-helper/live-lease design stale. Prefer replay+tail for **both** fresh and promoted fronts first; only revive a live-front continuation primitive if measured replay cost creates material target overhang. This removes a lifetime-sensitive API and keeps one MTP transition loop.

External scan: llama.cpp PR #29918 (`--cache-reuse-hybrid`) demonstrates a related upstream direction: preserve/reconstruct recurrent state when reusing surviving prompt text on hybrid recurrent/attention models. It is draft and not an MTP implementation, so do not import it; use it only as evidence that recurrent state ownership must be explicit rather than treating token/KV reuse as sufficient. Current llama.cpp release feed is b11390 (2026-10-04).

2026-10-05 first hardware screen (1321+1322, v6, 240K f16, quick ABA, same binary, BIGCHERRY_MTP_AHEAD=1 middle arm). Greedy identical in all arms at 24K and 80K. 24K: 74.4/76.1 base vs 75.7 t/s ahead; ms/step 41.5/41.0 -> 38.4; promoted 25/64 rounds (39%, = Gate 0 p_hit); ahead host ~16 ms per round, hidden under target verify. 80K: 53.5/54.6 vs 54.5 t/s; ms/step 50.4/49.9 -> 47.4; promoted 16/64 (25%); ahead ~21 ms. Per-step saving matches the skipped serial draft (~p_hit x draft time) but t/s is neutral because promoted fronts are accepted less (24K 168/257 vs 173/242; 80K 157/291 vs 161/279): a promoted token is MTP chain position 5-7 on draft hidden states, vs fresh drafts reseeded from target h every round. Next: promote only high-confidence tail tokens (stricter p_min for the tail, e.g. 0.8-0.9) or cap promotion to 1-2 tokens; record per-position acceptance of promoted vs fresh fronts to pick the cut.

2026-10-05 tail confidence cut (BIGCHERRY_MTP_AHEAD_PMIN, queue-ahead-pmin) at 24K: 0.85 -> 67.9 t/s vs 73.9/75.0 base (44.4 vs 41.1 ms/step), promoted 19/64 with ~2.5-token fronts; 0.95 -> 63.8 vs 75.0/75.2 (45.6 ms/step), 17/64, ~2.2 tokens. The cut shortens promoted fronts, and a short promoted front replaces a full fresh 3-token draft: worse than no cut. Greedy also differed from base in both cut arms (not without the cut); verify-batch-shape numerics are the likely cause but treat as unresolved. Keep default no cut. Next options: promote only full-length fronts (else fresh draft), or keep promotion and also prepare a target-seeded fresh draft in the overlap window.

2026-10-05 schedule (owner go-ahead): (1) re-run the 1321 + 1322 screen on the current pin and baseline (b11402 with 1333; every earlier number is from pin 0504396) - tools/lab/flash-next/queue-ahead-screen.sh b11402 2, two ABA sets per depth; (2) per-position acceptance telemetry for promoted vs fresh fronts; (3) promotion policy - note 1322 already promotes only full-length tails, so the remaining option from the list above is preparing a target-seeded fresh draft in the overlap window, or accepting the per-step gain where acceptance holds; (4) repair 1268 before FMTP05; (5) FMTP05 then FMTP07. The greedy difference seen in the confidence-cut arms stays unresolved and needs the QFP28 multi-request text gate before FMTP07.

2026-10-05 re-run on pin b11402 (build b-ahead-b11402 = deploy-v6-plus-ahead incl. 1333; queue-ahead-screen.sh): BROKEN at run time with BIGCHERRY_MTP_AHEAD=1 - llama-server aborts in llama_context::output_reserve, llama-context.cpp:2321 GGML_ASSERT(n_outputs_max <= cparams.n_outputs_max), reached from server_context_impl::decode (24K, first look-ahead arm; run /mnt/data/bigcherry-work/runs/ahead-b11402-d24576-r1/new/timing.server.log). The same binary with look-ahead off is normal (42.2 / 42.3 ms/step, 173/242 and 176/233). 1321 + 1322 still apply and compose cleanly (37/37), so the pin-bump tooling could not see this; it worked on pin 0504396. Upstream sizes each context's output buffer from the speculative depth (server_output_limits -> common_speculative_get_output_limits; draft context n_outputs_max = n_parallel in common/speculative.cpp), and the look-ahead path requests more output rows than that. Not yet determined: which context trips it (target verify of a promoted front, or the drafter's forced-front + tail decode) and which upstream commit introduced or tightened the limit. Fix belongs in the patch that owns the behaviour (1322 enables look-ahead; raise the relevant limit when BIGCHERRY_MTP_AHEAD is on). Production is unaffected: both patches are experimental and off by default. Steps 2-5 of the schedule above are blocked on this.

## Change Log

- 2026-10-04T06:32:25.863452+00:00: Added initial same-thread overlap design.
- 2026-10-05: Consolidated FMTP03 onto landed 1321 forced-front/tail primitive; removed planned duplicate helper/lease architecture and added slack-bounded tail policy.

## Ledger-events


- chg_20261004_161430_experimental-the-mtp-drafter_7189
- 2026-10-04T16:14:37.089254+00:00 (updated-by): Updated: section:ledger-events
- chg_20261004_161448_experimental-the-mtp-drafter_6717
- 2026-10-04T16:14:54.714881+00:00 (updated-by): Updated: section:ledger-events
- 2026-10-04T17:00:27.817054+00:00 (updated-by): Updated: section:notes
- 2026-10-04T17:31:34.739948+00:00 (updated-by): Updated: section:notes
- 2026-10-05T10:04:16.532717+00:00 (updated-by): Updated: section:notes
- 2026-10-05T10:30:54.194436+00:00 (updated-by): Updated: section:notes
- chg_20261005_113349_the-fused-decode-kernels-now-a_9568
- 2026-10-05T11:33:55.911140+00:00 (updated-by): Updated: section:ledger-events
