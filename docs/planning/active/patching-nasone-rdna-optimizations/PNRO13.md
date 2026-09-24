---
id: PNRO13
order: 0
plan: patching-nasone-rdna-optimizations
state: pending
created-at: '2026-09-09T10:53:08.931192+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: P1
---

# Direct pread-based lazy PLE table rows

## Description

TODO, NOT-READY (no files/functions/anchors specified, per GPT review). Evaluate direct pread-based lazy PLE table rows -- CORRECTED: verified at b11126 that lazy PLE is mmap-backed, not loader-callback-based: `TENSOR_READ_LAZY` drives `llama_model_loader::lazy_read::add()` (src/llama-model-loader.cpp:1088) and `lazy_read::buft()` (src/llama-model-loader.cpp:1080), and `load_all_data()` leaves the tensor (e.g. per_layer_tok_embd) mapped; at runtime, `ggml_get_rows` (CPU compute) directly dereferences the mmap'd pointer. There is NO existing loader callback seam where a pread implementation can simply be substituted -- this is a materially bigger design gap than the item previously implied.

## Steps

1. Design the runtime hook explicitly, not as a drop-in loader callback: the real integration points are `src/models/qwen4exp.cpp::build_inp_ple()` (verified real function, line 1185, called at line 383) on the graph-construction side, and CPU `ggml_compute_forward_get_rows()` (verify exact function name at implementation time) on the execution side -- OR a dedicated buffer/backend abstraction that carries file-offset metadata instead of a raw mmap pointer, since GET_ROWS today assumes direct pointer dereference into mapped memory.
2. Also audit `llama_model_loader::lazy_read` (src/llama-model-loader.h:89, src/llama-model-loader.cpp:1080/1088) for where a pread-based descriptor/offset table could be threaded through instead of (or alongside) the mmap path.
3. Specify descriptor ownership (which struct owns the open file descriptor for pread mode), row-offset metadata (how a GET_ROWS row index maps to a file byte offset without going through the mmap'd tensor pointer), and the mmap fallback path -- all currently unspecified -- BEFORE authoring the patch package.
4. Resolve PR #28136 and pin ancestry before local port.
5. Measure mmap lazy-read faults, bytes, latency and storage queue on a large PLE workload.
6. If absent upstream, implement explicit Linux pread mode with independent descriptor, Windows fallback and POSIX_FADV_RANDOM where available.
7. Sort/deduplicate requested rows while restoring original order and duplicates; support all quantized to_float types and direct F32 copy.
8. Bound workers and handle EOF/EINTR/invalid rows/descriptor failures without leaks; fall back to mmap lazy mode.
9. Compare cold/warm NVMe/cache-hot conditions; keep auto behavior unchanged until multiple storage environments prove benefit.

## Detailed Solution & Technical Design

The optimization exploits the ubatch sparse row set but trades page faults for userspace staging/dequant and threads. File offsets come from loader metadata, never tensor pointer arithmetic or mmap addresses.

## Code Samples & Guidance



## Files

Lazy-mode API/args; model/file loader; direct pread reader; Qwen4exp integration; platform fallback; row/error/thread tests; cold/warm I/O evidence.

## Validation

Exact row content/order/duplicates across supported types; EOF/EINTR/invalid/reopen failures; descriptor lifetime; worker cleanup; mmap fallback; cold/warm storage and model parity.

## Effort & Risk



## Standards

Upstream-first; deterministic row correctness; fail-safe fallback; no filesystem-cache-only performance claim.

## Acceptance Criteria

Staged rows equal mmap/reference; failures fall back or fail explicitly without leaks; declared large-PLE storage conditions show repeatable improvement; auto mode remains unchanged otherwise.

## Notes

Supersedes: NRO14
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-nasone-rdna-optimizations-nro14

2026-09-24 relevance at b11126: TODO. Grounded directly against b11126 source (TENSOR_READ_LAZY flag, lazy_read struct in src/llama-model-loader.{h,cpp}); anchors verified by grep, not guessed. GPT design request was submitted (dev-gpt-agent, batched with PNRO14/PNRO15) but the gateway queue was saturated (concurrent-session limit) and did not return a response within this session's time budget -- design was completed directly from source instead of via GPT synthesis.

2026-09-24 GPT review req_215c89d0b13a4bb7 applied: verified lazy PLE is mmap-backed via llama_model_loader::lazy_read (src/llama-model-loader.h:89, .cpp:1080/1088) with no existing loader-callback seam for pread substitution. Added the required runtime-hook design step around the verified real functions build_inp_ple (qwen4exp.cpp:1185/383) and CPU GET_ROWS execution, plus explicit descriptor-ownership/row-offset-metadata/fallback specification requirements before any package is authored.

## Change Log

- 2026-09-09T10:53:08.931192+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:09:37.677585+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.107936+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.765926+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:46:04.398329+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria
- chg_20260910_024630_the-remaining-nasone-successor_5195
- 2026-09-10T02:46:30.060469+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T02:30:06.821967+00:00 (updated-by): Updated: section:notes
- 2026-09-24T04:51:29.533411+00:00 (updated-by): Updated: section:description, section:steps, section:notes
