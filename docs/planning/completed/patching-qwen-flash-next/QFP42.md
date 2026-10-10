---
id: QFP42
order: 42
plan: patching-qwen-flash-next
state: completed
created-at: '2026-10-08T06:11:00+11:00'
breadth: ''
skill: advanced
created-by: agent
priority: P0
work: M
---

# MTP asynchronous NextN staging: remove the per-chunk target synchronization barrier

## Description



## Steps

### Step 0 (2026-10-08): source finding + blocked-time trace

24K target-context host timing (512-token chunks):

| arm | graph ms | set_inputs ms | graph_compute ms | CPU split compute ms | Meta input/compute ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| deferred ON | 17.6 | 21.2 | 56.5 | 12.4 | 11.9 / 32.2 |
| deferred OFF | 17.4 | 4.3 | 48.3 | 4.3 | 11.8 / 32.1 |

Source-verified at pin b11474 / production composition (line numbers below are after the production patch set is composed):

- The public `llama_get_embeddings_nextn()` is a real target barrier: `src/llama-context.cpp:4289-4292` calls
  `ctx->synchronize()`; `llama_context::synchronize()` reaches `ggml_backend_sched_synchronize()`
  (`src/llama-context.cpp:916-922`). 1348 calls that getter only after catch-up of the prior snapshot and before
  returning to the server.
- The +17 ms is **not** the 1326 input-sync mechanism. 1319's `inputs_us` encloses only
  `llm_graph_result::set_inputs()` (`src/llama-context.cpp:1611-1618`). 1326 modifies
  `ggml_backend_sched_copy_input()` (composed start 1830; its producer sync is 1847), which runs later from
  `ggml_backend_sched_compute_splits()` (composed start 1896; copy calls 1925/1930) and therefore belongs to 1319
  `compute_us`.
- 1348's snapshot is not the target input staging allocation. `bc_deferred_chunk::h_nextn` is its own
  `std::vector<float>`; llama's NextN output lives in `buf_output`, allocated from the output device's host
  buffer type and exposed as `embd_nextn` (`src/llama-context.cpp:2336-2387`). There is no alias to scheduler
  input staging.
- The draft CPU backend cannot still be executing its catch-up graph after `llama_process(ctx_dft)` returns:
  CPU backend graph compute calls synchronous `ggml_graph_compute()`
  (`ggml/src/ggml-cpu/ggml-cpu.cpp:170-190`). The final draft GPU split may still be asynchronous, but that is
  not a busy CPU thread-pool dependency.
- The next target submit cannot be waiting for the previous target stream: the synchronizing NextN getter above
  completes the target context before 1348 returns from `process_deferred()`.

Therefore none of the four proposed dependency waits explains the +17 ms inside 1319 `set_inputs`. The measured
+17 ms set_inputs and +8 ms CPU-split deltas are real host-time deltas, but source alone only supports an inference
that deferred ordering perturbs host/cache/memory/runtime conditions; it does not identify a removable dependency.
Do **not** book ~25 ms/chunk as recoverable QFP42 gain.

First measurement after PR #8: use fixed 1346 counters to separate synchronizing NextN wait, snapshot memcpy, interior
catch-up, terminal flush, and submit-to-submit interval on the same 24K/98K ABBA. If the +8 ms CPU split persists,
add per-input `set_inputs` timing and CPU split cycle/wall timing before changing scheduling. Only a demonstrated
non-dependency becomes an implementation step.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Done as 1359_prefill_pipeline (promoted 2026-10-11, #135-#138; QFP50 holds the design notes and measurements). Build b-pipe1 = production + 1359, one binary, BIGCHERRY_PREFILL_PIPELINE=1 against off, Flash-Next production profile:
- complete separation at 24K and 98K, and at 8K: prefill 1,380.2 / 1,393.4 against 1,306.7 / 1,309.7 t/s (8K), 1,437.5 / 1,429.1 against 1,261.3 / 1,328.4 (24K), 1,377.7 / 1,373.5 against 1,226.4 / 1,275.6 (98K); four requests at 24K +12.8% pooled; +15.7% against the native hook order;
- greedy target identity at every depth, twelve runs at 98K, prompts of 418 and 1,186 tokens;
- decode and acceptance unchanged; no error lines; activation marker BIGCHERRY_PATCH_HIT patch=1359_prefill_pipeline in the on arm only;
- cancel smoke (run cancel1, tools/lab/flash-next/long-ctx-profile.sh CANCEL_AFTER=8): a 30K-token prompt dropped after 8 s (server: `cancel task`, stopped at n_tokens = 12288), then the usual 24K request: text identical to the run without the cancel (md5 a789c151), no error lines, prefill 1,480.2 / 1,471.6 t/s;
- timing: kernel-gap-stats.py shows the once-a-batch gap gone (gaps of 10 ms or longer 81 -> 13 a card).
Not run: a context shift (needs a prompt beyond the context); it goes through the same reset_deferred path as the cancel.

## Effort & Risk



## Standards



## Acceptance Criteria

- The server prepares and submits target chunk k+1 before waiting for chunk k's NextN rows, and waits for chunk k only: met by 1359 (ggml backend event behind a second pinned copy of the rows; no host-wide synchronise on that path).
- Target semantics, KV, logits, MTP row order and catch-up order unchanged: met (greedy identity in every run).
- The gates listed under Gates: met as recorded under Validation, except the context-shift smoke, which was not run.
- The control flag returns to 1348's behaviour: met (flag unset = 1348's path, the off arm of every ABBA).

## Notes

2026-10-08: the retired QFP08 draft's other proposals are already covered - draft-during-verify overlap by 1322 (decode +4-6%), double-buffered hidden-state snapshot for prompt chunks by 1348 (prefill +5-10%), event-scoped NextN handoff by this item. Its promotion thresholds (>=5% or stop) are NOT carried over: owner policy is that small wins count.

2026-10-11 closure: done as 1359_prefill_pipeline (QFP50 has the measurements). The follow-ups this item had absorbed on 2026-10-08 are not part of that and were moved out on the owner's instruction: PRBE56 (scheduler split plan reuse) is QFP51, PRBE57 (NextN placement) is QFP52, and the one live idea from PRBE07 (kernels for verify widths 2..8) is noted in QFP38.

## Step 0, first data (Brutus 2026-10-08, PR #8 build 31a07866, Flash-Next 24K, 76 chunks of 512 tokens, target context)

| | graph | set_inputs | graph_compute | CPU split compute | Meta split input / compute |
|---|---|---|---|---|---|
| deferred catch-up on | 17.6 ms | 21.2 ms | 56.5 ms | 12.4 ms | 11.9 / 32.2 ms |
| deferred catch-up off | 17.4 ms | 4.3 ms | 48.3 ms | 4.3 ms | 11.8 / 32.1 ms |

With deferral the target's own host-side submission is about 25 ms per chunk slower (set_inputs +17 ms, CPU split +8 ms). Off path totals from 1346: target_block 25234.6 ms, draft_catchup 2210.2 ms, host gap 359.6 ms per chunk over 77 chunks. Not yet known: whether the extra set_inputs time is real contention (catch-up k holding the 1326 async host-input sync, a shared staging buffer, or the CPU backend pool that also serves token_embd) or the previous chunk's GPU time surfacing as a wait in a different place. 1346 prints zeros on the deferred path (chunks=0) and must be fixed before the per-call-site numbers can be read for that path. Cross-model: on Qwen3.8-27B built-in MTP 1348 is worth +0.8-1% prefill only, so the remaining gap is specific to a draft on its own card.

## Problem

QFP31 / patch 1348 defers MTP draft catch-up by one target chunk. Brutus b11474 adoption ABBA improved prefill
+5.4% / +6.3% / +9.7% at 8K / 24K / 98K with complete separation and identical greedy target output, but this is
well below the ~19% whole-gap upper bound measured before implementation.

The remaining serialization is visible in 1348's ordering. After target chunk k+1 is submitted,
`common_speculative_process_deferred()` first runs draft catch-up k, then calls the public
`llama_get_embeddings_nextn(ctx_tgt)` for k+1. That API synchronizes the target context before returning the already
enqueued host NextN rows. Only after it returns can the server prepare target chunk k+2. The 2026-10-08 source audit
also shows that the observed +17 ms `set_inputs` and +8 ms CPU-split host deltas are not that target barrier and are
not yet attributable to a removable dependency; they are excluded from QFP42's gain budget until measured.

The 83 ms / 437 ms estimate was the whole MTP-induced gap, not 83 ms of draft compute. Prior timing measured actual
draft decode at roughly 29-47 ms/chunk (6.6-9.9% of prompt wall), which matches QFP31's observed gain. The residual
lever is the blocking NextN acquisition plus the host preparation it keeps behind the synchronization barrier.

## Goal

Make the target's MTP NextN rows for chunk k available through an MTP-owned asynchronous staging slot without a
host-wide target synchronization. The server must be able to prepare and submit target chunk k+1 before waiting for
k's staged rows. After k+1 is submitted, wait only for k's staging completion and run draft catch-up k while k+1
computes.

Target semantics, target KV, target logits, sampling, MTP row order, and draft catch-up order must remain unchanged.

## Proposed ordering

Native/QFP31 current interior loop:

1. submit target k;
2. run catch-up k-1 under target k;
3. blocking `llama_get_embeddings_nextn()` waits for target k;
4. copy NextN k to host;
5. return to server;
6. build/set inputs/rebuild/launch target k+1.

QFP42 target ordering:

1. submit target k;
2. enqueue NextN-k device-to-pinned-host staging behind target-k compute on the owning target stream(s), recording a
   completion event; do not synchronize the host;
3. return to server immediately;
4. build/set inputs/rebuild/launch target k+1;
5. wait for staging event k only;
6. run draft catch-up k while target k+1 computes;
7. repeat.

The final staged chunk and final catch-up are flushed before `common_speculative_begin()` / sampling.

## Implementation constraints

- Build on promoted 1348 after PR #13 merges; do not duplicate its snapshot/cancellation state machine.
- Add two MTP-owned staging slots matching 1348's two deferred chunks. A slot owns token metadata, pinned host storage,
  backend/event handles, row count and generation/validity state.
- Do not expose a raw pointer whose backing target output can be reused by the next target submit.
- For a non-Meta single backend, enqueue the copy on the backend stream after the graph's NextN producer.
- For Meta/tensor split, staging must follow the actual output ownership. If NextN is mirrored, copy one authoritative
  mirrored output. If split/partial, gather/copy the required row parts using Meta's per-device ownership rather than
  calling `ggml_backend_meta_buffer_get_tensor` as if it were a simple buffer.
- Prefer an explicit narrow API such as `llama_embeddings_nextn_stage_async(ctx, dst, event)` / wait, owned by the
  llama context/output path. Do not make generic scheduler output lifetime globally longer.
- Pinned host storage is allocated/reused outside the per-chunk hot path; no per-chunk malloc/free.
- Staging failure, cancellation, context shift, cache load/restore, or target decode failure invalidates the slot and
  falls back to the existing target-only poison behavior from 1348.
- No worker thread in v1. Add one only if asynchronous staging still leaves measurable host serialization.

## Instrumentation

Extend the existing QFP31 timing path or add a narrow env-gated diagnostic with per-prompt totals:

- `nextn_stage_enqueue_ms` host enqueue time;
- `nextn_stage_wait_ms` host wait after next target submit;
- `nextn_stage_copy_ms` if backend events can measure device copy time;
- `draft_catchup_ms`;
- `final_flush_ms`;
- target submit-to-submit interval.

Activation marker: `BIGCHERRY_PATCH_HIT patch=<new-id> mechanism=async-nextn-stage`.

## Gates

### Offline

- mechanics test: every Edit exact once, idempotent, bounded span;
- patch-lint;
- compose current production + new patch at b11474 or current pin;
- tests exercise two-slot reuse, final flush, failure invalidation, cancel/context-shift poison, and Meta ownership path;
- control flag returns exactly to promoted 1348 behavior.

### Hardware

Brutus, Flash-Next IQ4_XS, production layout, ctx 245760, f16 KV. One binary ABBA:
A = async staging on, B = async staging off / promoted 1348 behavior.

At 8K / 24K / 98K record:
- prefill t/s and TTFT;
- target submit-to-submit interval;
- stage enqueue/wait/copy and draft catch-up totals;
- activation marker;
- decode t/s and MTP acceptance;
- greedy target md5.

Required:
- complete separation at 24K and 98K;
- no prefill regression at 8K;
- greedy target identity at all depths;
- no acceptance/decode material regression;
- no error lines;
- cancel/context-shift smoke clean.

Success target: recover a material part of the residual gap above QFP31. The theoretical ceiling is not treated as an
acceptance threshold; the timing counters must show that host preparation moved ahead of the NextN wait.

## Dependencies

- QFP31 / 1348_mtp_deferred_catchup promoted and merged.
- 1346 timing diagnostic retained without its unqualified prompt-window mechanism.

## Consolidated RDNA/MTP follow-ups (PRBE07, PRBE56, PRBE57)

These are additional **qualification/design steps**, not modifications of the 1348 deferred-catch-up semantics or the QFP42 async-staging acceptance gate:

1. **PRBE56 scheduler plan/sync:** use `ggml/src/ggml-backend.cpp::ggml_backend_sched_split_graph` and the per-split event synchronization path to profile split/build CPU time, waits blocked in target versus draft, bytes and copies/token for MTP verify widths 2..8. Cache only a stable (shape/topology, backend-ID mapping, split-boundary) **template** and rematerialize per-instance tensor pointers, `split->inputs`, and graph-owned views. Raw `ggml_backend_sched_split` reuse is unsafe because graph splitting mutates `node->src[]`. The existing `prev_backend_id != split_backend_id` check already avoids same-backend sync: never claim removing that existing guard is a new win. Only remove an actual cross-device wait proven non-dependent by the QFP42 blocked-wall-time trace.
2. **PRBE57 NextN placement:** the referenced external fork commit `1fcc05da` changes Vulkan NextN tensor placement; `41a8ca78` is a separate backend-resident hidden-state handoff (Vulkan PRBE58, parked). Neither is automatically a HIP implementation. In QFP42's HIP/meta production layout, measure the owner device of `NEXTN_PROJ_PRE/POST` and `t_h_nextn`, the real D2H/P2P copies, and a candidate owner cost of copy-in+copy-out+sync+host-staging. Consider `BIGCHERRY_MTP_TOPOLOGY=off|auto|device:N` in a separate default-off placement ablation only if QFP42 trace shows a copy/ownership bottleneck. Validate byte-identical NextN rows, greedy identity, accepted counts, copies/token and no extra VRAM pressure; do not conflate home-backend placement with async staging.
3. **PRBE07 BridgeSpec evaluation:** PRBE07 is now complete as **inspiration-only**, not an adoption. Main audited MIT-licensed BridgeSpec 0.1.0 at commit `2b846f2ff1eb95ac84e4b0488882b7e4066bff14`: Windows single-GPU/gfx1100, singleton host DLL, missing context-shift/multi-request lifecycle, no production interop semaphores and external-only speed evidence. Native BigCherry FMTP/QFP ownership is preferred. Do not implement a second sidecar; consider only a fresh measured width-2..8 kernel opportunity or vocab-head design through its owning patch plan, with first-party hardware and target-authoritative identity.

Decision gate: retain only profiled HIP-specific work that reduces blocked time or copies while keeping QFP42's existing target/draft correctness contract. PRBE07/56/57 are closed as planning duplicates **into this open item**; no patch promotion is implied.

## Change Log

- 2026-10-08 (step 0): Source audit ruled out 1326, shared staging, a still-running draft CPU graph, and the prior target stream as the cause of the +17 ms set_inputs delta; fixed 1346 timing is the next measurement gate.
- 2026-10-08 (triage): Folded PRBE07, PRBE56, PRBE57 into source-scoped MTP/NextN and BridgeSpec qualification notes.

- 2026-10-07T22:53:39.805664+00:00 (updated-by): Updated: section:steps, section:notes
- 2026-10-08T08:45:47.085065+00:00 (updated-by): Updated: section:notes
- 2026-10-10T21:11:47.064660+00:00 (updated-by): Updated: section:validation, section:acceptance_criteria
- 2026-10-10T21:11:54.676673+00:00 (state-transition): State: pending → completed
- 2026-10-10T21:24:08.031791+00:00 (updated-by): Updated: section:notes

## Ledger-events

- chg_20261010_213553_prompt-processing-with-a-separ_1650
- 2026-10-10T21:36:00.397315+00:00 (updated-by): Updated: section:ledger-events
