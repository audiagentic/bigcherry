---
id: THA26
order: 0
plan: tuning-hip-autotune
state: pending
created-at: '2026-09-09T10:50:43.973052+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: null
---

# Dispatch integration — use transformed winner at runtime with zero overhead when no transform applies

## Description

Integrate transformed tuning winners at dispatch with zero overhead on untransformed signatures and fail-closed runtime fallback.

## Steps

Carry transform_id/inverse mapping in resolved binding; apply transform before candidate launch and inverse writeback after; preserve fast path for transform_id=0; validate batch M MVF, cache warm path, can_execute fallback and bit-identical native output.

## Detailed Solution & Technical Design

Implement directional HI30→THA26 seam; extend replay binding/cache identity without circular dependency. Transform is optional and runtime applicability is rechecked; any failure selects native.

## Code Samples & Guidance



## Files

HIP autotune types/dispatch/cache and transform producer/consumer tests.

## Validation

End-to-end transformed binding, no-transform overhead, full batch output, can_execute fallback, process-cache warm path, native bit identity.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Transformed winners execute correctly with inverse writeback, untransformed dispatch remains behavior/performance equivalent, and any invalid transform fails closed to native.

## Notes

Supersedes: HI31
Migration: capability-rebaseline-v3-2026-09
Successor key: tuning-hip-autotune-hi31

## Change Log

- 2026-09-09T10:50:43.973052+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:07:02.978586+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:00.974607+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-09T12:09:27.149595+00:00 (updated-by): Updated: section:title
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.565652+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:33:33.368652+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria
- chg_20260910_033345_repaired-three-more-tuning-suc_6484
- 2026-09-10T03:33:45.461102+00:00 (updated-by): Updated: section:ledger-events
