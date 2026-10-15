---
id: QFP08
order: 0
plan: patching-qwen-flash-next
state: completed
created-at: '2026-10-03T15:20:57.065205+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P2
work: L
---

# Draft-prefill overlap — superseded by validated 1348 and 1359

## Terminal disposition (2026-10-14; BCOP215)

**SUPERSEDED_IMPLEMENTED.** QFP08's original goal—run draft catch-up for prompt chunk k while target chunk k+1 executes—was implemented and measured by validated `1348_mtp_deferred_catchup` and `1359_prefill_pipeline`. Do **not** create the QFP08 standalone pipeline, worker, ring, scheduler, event API, cache, patch package or hardware queue. QFP08 no longer owns an unresolved implementation. Its 2026-10-03/05 draft-prefill steps and the later decode-time sync/double-buffer notes are historical; decode-time handoff belongs to FMTP/QFP15/QFP27, not QFP08.

### Exact implementation and lifetime (pinned llama.cpp b11552 + validated BigCherry overlays)

- Native `common/speculative.cpp::common_speculative_impl_draft_mtp::process` pairs each token with the **previous** target hidden row; the first token of a chunk uses `pending_h[seq_id]`. It calls `llama_process(ctx_dft, ...)` for the draft catch-up and then copies target rows into `verify_h` and the next `pending_h`. It assumes each sequence's tokens form a contiguous group (source TODO; not a general interleaved-sequence pipeline).
- Native `src/llama-context.cpp::llama_context::decode` enqueues `t_h_nextn` extraction via `ggml_backend_tensor_get_async` into `embd_nextn`; `get_embeddings_nextn` returns that context-owned host storage after `output_reorder`. It does not itself create an immutable cross-chunk lease. The server's `decode` path synchronizes a target batch with outputs before calling `common_speculative_process`. A naive background consumer of `embd_nextn.data` can therefore read rows overwritten/reordered by a later chunk.
- Validated **1348** owns prompt-only deferred catch-up: it snapshots the current chunk's token metadata and shifted hidden rows into two MTP-owned host snapshots, submits the next target chunk, then runs the previous draft catch-up on the same host thread. It flushes the last pending chunk before generation and poisons/drops stale work on failure, cancel or context shift.
- Validated **1359** owns the remaining target-submit gap: a second, parity-selected pinned host copy of unmasked NextN rows is enqueued behind target work, with backend events on **every physical Meta child**. The server submits chunk k+1 before waiting only for chunk k's fenced rows; buffer reuse waits for the previous slot's events. No new worker thread, GPU P2P, cross-rank loan, scheduler or independent allocator is needed. Unsupported events/masked output take the synchronous fallback. 1326 retains stream-ordered input ownership.
- **Current switch truth:** 1348 is enabled unless `BIGCHERRY_MTP_DEFERRED_CATCHUP=0`; 1359 is enabled when `BIGCHERRY_PREFILL_PIPELINE` is **unset** and disabled by `=0` (source `patch.py`: `getenv(...) == nullptr || atoi(getenv(...)) != 0`). The older QFP50/1359 README statements that the pipeline is default-off describe its *initial promotion*, not the present patch. Do not label an unset arm as the off/control arm in new experiments.

### Existing BigCherry measurements (historical b11474, not a new b11552 run)

- 1348 versus native catch-up: prefill +5.4% at 8K, +6.3% at 24K and +9.7% at 98K on Flash-Next; greedy text identical and decode neutral in those runs.
- 1359 versus the then-default 1348-only arm: +6.0% at 8K, +7.4% to +10.7% at 24K and +10.0% at 98K; against the native hook order +15.7% at 24K in the recorded same-binary comparison. These are **different controls**, not additive percentages. 24K long kernel gaps >=10 ms fell from 81/2.78 s to 13/0.81 s; idle share 17.4% -> 12.9%. Historical 24K no-drafter prefill was +14.6% versus the old baseline, a contextual ceiling, not a new measured pipeline result.
- A Qwen3.8-27B built-in/shared-MTP negative control did not activate 1359 and was neutral. Recorded text identity included 12 runs at 98K. Current-pin and cancellation/context-shift qualification must not be inferred beyond the actual archived evidence.

### Residual correctness and performance ownership — no QFP08 reopening

1. `QFP50/1359` owns physical child-event fanout, fenced NextN storage, two-slot reuse, graph rebind and remaining prefill boundary gaps. `1348` owns snapshot, `pending_h`, final flush and cancel/shift state. `QFP27/1326` owns scheduler-input lifetimes. `FMTP02/03/1321/1322` owns **generation-only** look-ahead; `QFP15/1315` owns target-seed/draft provenance. Do not reimplement or independently benchmark their code paths under QFP08.
2. A further optimization is admissible only after an exact b11552 composed trace proves a **new, exposed** gap not already hidden by 1348+1359. Record target compute, NextN D2H, per-child event completion, draft catch-up, target-next-submit, final flush, host copies, PCIe traffic, graph UID, physical device and per-request/ubatch identity. On a 5120-wide F32 NextN row, a 512-token chunk is **10,485,760 bytes (10 MiB)** and 80K tokens imply **1,638,400,000 bytes** of hidden-row payload per direction in the worst uncompressed full-catch-up model; these are **arithmetic sizes**, not measured PCIe bytes or transfer time. Count both target D2H and drafter H2D where actually observed; never assume peer transfer on the gfx1030/1100/1201 no-P2P topology.
3. Correctness fence: upstream llama.cpp [#26827](https://github.com/ggml-org/llama.cpp/pull/26827) is **open** (checked 2026-10-14) and proposes synchronization for multi-ubatch MTP draft KV reuse; it is not in b11552. This is a *separate reported risk*, not a reproduced BigCherry fault. Do not weaken per-ubatch draft KV ordering while optimizing prefill. Merged [#30459](https://github.com/ggml-org/llama.cpp/pull/30459) adds a separate draft `n_ubatch` **after** b11552; do not silently mix that pin change into the control. For actual regressions require two requests in one process, multi-ubatch, 8K/24K/98K, cancel/shift, graph on/off, per-child delayed completion, full logits/KLD/greedy and MTP acceptance; missing draft/target work is failure.
4. Terminal outcomes: `ALREADY_IMPLEMENTED` (this item), `OWNER_RECONCILIATION` (route any future 1348/1359 defect to its existing owner), or `NO_NEW_EXPOSED_GAIN` (if measured residual <3% of E2E prefill/TTFT). Any future candidate needs a proved new gap, safe ownership and CI95-low >=3% prefill/TTFT, <=1% decode/control regression; no second QFP08 patch.

### External mechanism comparison

[SGLang-JAX EAGLE hidden-state flow](https://github.com/sgl-project/sglang-jax/blob/main/docs/architecture/09-speculative-decoding.md) explicitly distinguishes FULL hidden capture on target prefill from LAST-only draft decode, supporting the same **per-row ownership requirement**, not an AMD implementation port. [vLLM P-EAGLE](https://vllm-project.github.io/2026/03/13/p-eagle.html) parallelizes trained-head *generation* on NVIDIA B200, not target/draft prefill catch-up on separate no-P2P GPUs. Neither establishes further BigCherry gain.

### Audit and novelty receipt

Selected `feat/rad03-gpu-lanes` from independent `e4d95b98fadef068fc6e1058dec3ec61b29e768b` (2026-10-14 18:54 UTC; RAD05 opt-in helper GEMV). QFP08 last independent update `dde3dca5abffe26fd1f46a114eebd71d12a1d6ec` (2026-10-05), 1348/1359 last relevant independent updates 2026-10-11. No QFP08/1348/1359 independent commit, PR submission, hardware queue or experiment in the prior 12 hours was identified. Excluded active RAD02/RAD03/RAD04/RAD05 helper, mailbox, routing, placement and prefill lanes; recent BCOP205–214 slices cover other mechanisms (BCOP213 is draft **trace attribution**, not prefill pipeline ownership).

Pinned upstream source checks: **10/10 passed** (two whitespace-tolerant checks corrected after an initial 8/10 literal-text run). Host-only handoff oracle: **7/7 passed**. No repository pytest, patch application, HIP build, GPU run or new inference benchmark.

## Historical provenance

Original 2026-10-03 QFP08 described an unimplemented pipeline and projected 5–30% of draft-related TTFT without hardware qualification. 2026-10-04 FMTP split moved generation handoff to FMTP; 2026-10-05 BCOP20 backfill appended decode notes that do not reopen QFP08. Subsequent validated 1348 and 1359 implemented the prompt pipeline; QFP50 was completed 2026-10-11. This item is now completed to avoid a duplicate scheduler/event implementation.
