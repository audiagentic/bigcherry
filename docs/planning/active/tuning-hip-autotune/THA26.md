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

**Transferred THA28 acceptance (independent gate):** After the existing replay-v5 foundation (do not backport v4 markers), extend THA26's runtime resolved binding/cache with opt-in, C++/Python byte-identical generalized masked digests for MMQ family/type/K/M and MMVQ family/type/full-ne0 only; explicitly forbid MMVF/BLAS generalized keys. Export a representative *promoted* highest-call winner with collision assertion and immutable candidate/provenance link, look up exact -> generalized -> native, record match_kind and independent exact/generalized/fallback counters, guard each generalized recommendation using current can_execute/blacklist/executor, and preserve exact replay behavior when disabled. Run positive/negative alignment-class held-outs, corrupt/unknown identity and transformed winner cases under graph capture, with launch proof and native fallback. Generalized key never authorizes an otherwise illegal candidate or extra transform, and the zero-overhead transform_id=0 native fast path stays unchanged.

## Notes

Supersedes: HI31
Migration: capability-rebaseline-v3-2026-09
Successor key: tuning-hip-autotune-hi31

CONSOLIDATION 2026-11-22 HIP-AUTOTUNE: receiving THA28. Existing scope and acceptance are retained. The original historical source plans and evidence are archived as superseded, not completed. No new kernel, orchestrator, scheduler, patch, GPU result, runtime activation or performance qualification is implied. Changes use the existing owning implementation only.

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

## Reviews

- RV4501
- 2026-11-22T10:18:28.226474+00:00 (updated-by): Updated: section:acceptance_criteria, section:notes
