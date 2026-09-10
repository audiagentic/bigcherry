---
id: PNRO05
order: 0
plan: patching-nasone-rdna-optimizations
state: pending
created-at: '2026-09-09T10:52:33.316789+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: P0
---

# Chunk GDN MTP prefill prefix while preserving snapshot tail

## Description

Qualify the MTP-specific GDN prefix/tail composition independently from PNRO04's raw kernel. Chunk only the long prefix and preserve the final K snapshot-producing recurrence.

## Steps

1. Require PNRO04 and enable only non-KDA, K>1, n_seqs==1, sufficiently long supported sequences.
2. Compute n_prefix=n_tokens-K, run chunked prefix with temporary pool-owned state, and use existing sequential recurrence for exactly final K tokens.
3. On synchronous rejection/failure, use original full-sequential path.
4. Validate every snapshot slot, final recurrent state, output/logits, threshold edges, K=2/3/5/8, sequence count, graph capture/replay, and pool lifetime.
5. Include FP32 chunked-prefix control where feasible to separate orchestration errors from BF16 kernel error.
6. Measure prefix savings versus transition/allocation overhead and MTP acceptance.

## Detailed Solution & Technical Design

Capability owner: patching

Split assessment: One independent boundary; Build/Run support is a dependency.

Overlap assessment: No duplicate boundary found; related items are prerequisites or adjacent evidence.

## Code Samples & Guidance



## Files

patches/1254_nro05_gdn_mtp_prefix_tail; orchestration/snapshot path; snapshot fixtures; MTP evidence

## Validation

Per-slot and final-state parity, continuation behavior, nonqualifying-shape fallback, threshold±1, K extremes, single/multi-sequence, launch rejection, pool reuse, graph replay, and acceptance-aware E2E performance.

## Effort & Risk



## Standards

Snapshot/state restoration correctness; preserve sequential fallback; preregister tolerances inherited from PNRO04.

## Acceptance Criteria

Every snapshot and final state matches registered tolerance; all nonqualifying shapes retain sequential path; MTP acceptance is not regressed; positive E2E effect is statistically supported.

## Notes

Supersedes: NRO05
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-nasone-rdna-optimizations-nro05

Supersedes: NRO05
Inherited semantic scope: preserve prefix/tail boundary, snapshot ordering, failure fallback, state lifetime, and acceptance-aware qualification.
Migration: capability-rebaseline-v3-2026-09

## Change Log

- 2026-09-09T10:52:33.316789+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:08:52.793774+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.071633+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.713426+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:26:22.478825+00:00 (updated-by): Updated: section:description, section:steps, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes
- chg_20260910_022800_five-nasone-successor-plans-no_4030
- 2026-09-10T02:28:00.303796+00:00 (updated-by): Updated: section:ledger-events
