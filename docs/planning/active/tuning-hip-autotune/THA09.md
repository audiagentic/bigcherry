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

Complete the residual lazy-native-select experiment: the memoized provider mechanism and force-once/per-site counters are already implemented; only a reachable guard-deferral experiment and end-to-end proof remain.

## Steps

1. Preserve the implemented zero-allocation memoized provider and existing behavior with both guards.
2. Add an explicit experiment mode that moves BOTH native.valid guards (dispatch_try and resolve) only when enabled.
3. Collect a non-vacuous shadow validity comparison in that mode; do not claim evidence from the guarded path.
4. Run correctness, graph-capture, diagnostics-OFF, and end-to-end latency/no-regression gates.
5. Keep the default path unchanged unless the measured sub-1% policy and evidence gates pass.

## Detailed Solution & Technical Design

Capability owner: tuning

Split assessment: One independent boundary; Build/Run support is a dependency.

Overlap assessment: No duplicate boundary found; related items are prerequisites or adjacent evidence.

## Code Samples & Guidance



## Files

successor-specs/tuning-hip-autotune-hi158.md

## Validation

Historical evidence and constraints: Preserve frozen notes/reviews/evidence on predecessor; IDs: HI87.

Active dependencies: Frozen dependencies: none recorded.

Reference handling: Rewrite forward references (1); preserve historical references (1) on predecessor.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

The experiment can observe real guard-deferral behavior, proves correctness and graph capture, and demonstrates an attributable end-to-end result within the sub-1% policy before any default promotion.

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
