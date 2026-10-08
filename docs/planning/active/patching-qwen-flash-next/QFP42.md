---
id: QFP42
order: 42
plan: patching-qwen-flash-next
state: pending
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



### Step 0 (added 2026-10-08): blocked-time synchronisation trace before any code

Salvaged from the retired QFP08 draft (branch automation-qfp-indexer-20261004, deleted 2026-10-08). Extend the sync tracer (`tools/lab/flash-next/` synctrace mode of `long-ctx-profile.sh`, `sync-tracer.c`) from call counts to blocked wall time per call site, tagged with context (target / draft), prompt chunk index and whether the wait is inside `llama_get_embeddings_nextn`, `common_speculative_process_deferred`, draft graph compute or `llama_synchronize`. Run it on the 1348 build at 24K and 98K with deferred catch-up on and off. Output: a table of ms blocked per chunk by call site that accounts for the gap between the measured +6-10% and the ~19% idle-gap bound (83 ms of 437 ms per chunk). Only waits shown there to be non-dependencies are removed in the later steps; a wait that is a true data dependency is recorded as such and left alone.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation



## Effort & Risk



## Standards



## Acceptance Criteria



## Notes



2026-10-08: the retired QFP08 draft's other proposals are already covered - draft-during-verify overlap by 1322 (decode +4-6%), double-buffered hidden-state snapshot for prompt chunks by 1348 (prefill +5-10%), event-scoped NextN handoff by this item. Its promotion thresholds (>=5% or stop) are NOT carried over: owner policy is that small wins count.

## Problem

QFP31 / patch 1348 defers MTP draft catch-up by one target chunk. Brutus b11474 adoption ABBA improved prefill
+5.4% / +6.3% / +9.7% at 8K / 24K / 98K with complete separation and identical greedy target output, but this is
well below the ~19% whole-gap upper bound measured before implementation.

The remaining serialization is visible in 1348's ordering. After target chunk k+1 is submitted,
`common_speculative_process_deferred()` first runs draft catch-up k, then calls
`llama_get_embeddings_nextn(ctx_tgt)` for k+1. That getter synchronizes the current target before copying the NextN
rows. Only after it returns can the server prepare target chunk k+2 (graph build, set_inputs, CPU split, Meta subgraph
rebuild and launches). Therefore QFP31 hides draft compute but still prevents host preparation of the following target
chunk from overlapping target GPU compute.

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

- 2026-10-08 (triage): Folded PRBE07, PRBE56, PRBE57 into source-scoped MTP/NextN and BridgeSpec qualification notes.

- 2026-10-07T22:53:39.805664+00:00 (updated-by): Updated: section:steps, section:notes
