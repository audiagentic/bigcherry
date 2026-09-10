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

Evaluate direct pread-based lazy PLE table rows only after upstream ancestry check, with explicit platform fallback and deterministic row semantics.

## Steps

- Resolve PR #28136 and pin ancestry before local port.
- Measure mmap lazy-read faults, bytes, latency and storage queue on a large PLE workload.
- If absent, implement explicit Linux pread mode with independent descriptor, Windows fallback and POSIX_FADV_RANDOM where available.
- Sort/deduplicate requested rows while restoring original order and duplicates; support all quantized to_float types and direct F32 copy.
- Bound workers and handle EOF/EINTR/invalid rows/descriptor failures without leaks; fall back to mmap lazy mode.
- Compare cold/warm NVMe/cache-hot conditions; keep auto behavior unchanged until multiple storage environments prove benefit.

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
