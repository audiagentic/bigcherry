---
id: THA09
order: 0
plan: tuning-hip-autotune
state: pending
created-at: '2026-09-09T10:49:14.603876+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: P1
---

# Lazy native_select on the dispatch hot path (reopens HI87 under the sub-1% policy)

## Description

Evaluate lazy native_select on the HIP dispatch hot path under the strict sub-1% policy, preserving force semantics and structural proof.

## Steps

Implement zero-allocation memoized native provider keyed by tensor pointers/context with computed flag/cache; make dispatch_resolve consume provider; preserve force-once semantics and do not move !native.valid guard; add counters for dispatch/L1 hits/misses/native calls/forces; compare baseline and patched structural counts and interleaved E2E.

## Detailed Solution & Technical Design

Avoid computing native selection on ~99.8% of dispatches where cache already resolves. No std::function/allocation. L1-hit path must never force. Counter reduction proves mechanism but does not justify shipping absent E2E benefit.

## Code Samples & Guidance



## Files

HIP native_select/dispatch resolve; structural counters and diagnostics; hot-path tests and Brutus benchmark evidence.

## Validation

native_select calls approximate misses plus force paths; correctness/force behavior unchanged; end-to-end and dispatch overhead measured with interleaved controls.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Ship only if zero-allocation behavior is proven and end-to-end regression is below the explicit sub-1% policy; counter improvement alone is insufficient.

## Notes

Supersedes: HI158
Migration: capability-rebaseline-v3-2026-09
Successor key: tuning-hip-autotune-hi158

Supersedes: HI158
Inherited constraint: RV133 — do not reimplement completed mechanism; both validity guards must be named and the shadow counter must be reachable/non-vacuous.
Migration: capability-rebaseline-v3-2026-09

## Change Log

- 2026-09-09T10:49:14.603876+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:05:16.500617+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:00.867999+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.389969+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T00:52:57.320952+00:00 (updated-by): Updated: section:description, section:steps, section:acceptance_criteria, section:notes
- chg_20260910_005948_legacy-planning-folders-now-co_1240
- 2026-09-10T00:59:49.044583+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:30:11.111314+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria
- chg_20260910_033046_repaired-four-more-tuning-succ_3978
- 2026-09-10T03:30:46.991974+00:00 (updated-by): Updated: section:ledger-events
