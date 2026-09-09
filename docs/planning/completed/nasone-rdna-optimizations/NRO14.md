---
id: NRO14
order: 14
plan: nasone-rdna-optimizations
state: superseded
created-at: '2026-09-08T09:50:40+10:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: L
---

# Direct pread-based lazy PLE table rows

## Description

Track nasone commit `abca85cdc608efe8edc89546e1c3e5d6323b6308`, sourced from upstream PR #28136. It adds `--lazy-mode on-direct` for very large per-layer embedding/PLE tables: known row IDs are sorted/deduplicated, fetched with explicit `pread()` from an independently opened descriptor, optionally dequantized to F32, and staged for graph use instead of relying on mmap demand paging.

As with NRO13, ancestry to the current pin must be checked before any local backport.

## Steps

1. Resolve #28136 upstream state and pin ancestry.
2. Reproduce mmap lazy-read page-fault/I/O behavior on a model with large PLE tables; capture page faults, bytes read, latency, and storage queue behavior.
3. If still absent upstream, port direct mode with Linux-specific `pread`, Windows fallback, independent file descriptor, and `POSIX_FADV_RANDOM` where supported.
4. Validate row sorting/dedup preserves requested output order and duplicate rows.
5. Validate all quantized row types supported by `to_float`; F32 path copies directly.
6. Bound worker count and handle thread-creation/read exceptions without leaked joinable threads.
7. Test EOF, EINTR, invalid row, non-reopenable FILE-backed model, and fallback to mmap lazy mode.
8. Compare cold/warm storage, NVMe and cache-hot conditions; direct reads may not dominate everywhere.

## Detailed Solution & Technical Design

This is an I/O optimization, not GPU tuning. The key advantage is explicit knowledge of the entire ubatch's sparse row set: the reader can issue only required rows, deduplicate repeats, avoid sequential readahead, and control concurrency. The cost is userspace staging/dequant plus thread management.

File-offset identity must come from model-loader metadata. Do not derive offsets from tensor pointer arithmetic or mmap addresses. The independently opened descriptor avoids changing flags/advice on the loader's file description.

## Code Samples & Guidance

Keep direct mode explicit during qualification. Do not change `auto` semantics until multiple storage environments are characterized.

## Files

Planning-only until ancestry check; likely common args, llama API enum, file/model loader, lazy reader implementation, Qwen4exp model integration, and I/O tests.

## Validation

Row-content equality across types/duplicates, error handling, fallback, descriptor lifetime, cold/warm I/O measurements, model output parity.

## Effort & Risk

High portability/I/O complexity but isolated from GPU arithmetic.

## Standards

Upstream-first, deterministic row correctness, fail-safe fallback, no performance claim from filesystem cache alone.

## Acceptance Criteria

- Exact staged rows equal mmap/reference for all supported types.
- Failures fall back or fail explicitly without thread/resource leaks.
- Real large-PLE workload shows repeatable latency/throughput improvement in declared storage conditions.

## Notes

Do not combine with QSA gather despite both targeting Qwen4exp; they optimize independent I/O versus attention costs.

Superseded by: PNRO13
Migration: capability-rebaseline-v3-2026-09

## Change Log

- 2026-09-08T09:50:40+10:00 (created-by): Created from nasone/upstream #28136; P1.

## Ledger-events


- Pending: ag-ledger MCP unavailable in authoring session.
- 2026-09-09T11:25:30.701521+00:00 (updated-by): Updated: section:notes
- 2026-09-09T11:43:35.221434+00:00 (state-transition): State: pending → superseded
- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:57:59.908072+00:00 (updated-by): Updated: section:ledger-events
