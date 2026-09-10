---
id: PRBE52
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:57:05.575180+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: null
---

# UP-MTP-001: Adaptive MTP draft depth

## Description

Implement and qualify the consolidated adaptive MTP draft-depth controller from RD62 and duplicate NRO06. This is one backend-neutral speculative-policy owner; NRO06 is not an independent implementation.

## Steps

1. Freeze source identity and re-audit ancestry after each pin bump; preserve the distinct adaptive type while leaving fixed draft-mtp unchanged.
2. Add n_min_adaptive/floor and cap configuration with explicit validation.
3. Implement per-sequence controller state: current depth, climb streak, drop pressure, reset semantics, and heterogeneous-sequence isolation.
4. Preserve source policy semantics: floor/cap, full-accept climb, miss-pressure drop, floor non-accumulation, and depth-3→4 barrier as hypotheses rather than universal constants.
5. Instrument requested/accepted depth transitions and support offline acceptance-trace replay; keep one pre-registered runtime policy during qualification.
6. Validate request/sequence/context-rewind/failure/retry/recreation resets and no state leakage.
7. Compare against fixed depths and production fixed depth over prose, code, repetitive, and reasoning-like content at multiple contexts; measure acceptance, rejected work, target/draft latency, and total TPS.
8. Require deterministic target-output parity and exhaustive controller tests before any promotion.

## Detailed Solution & Technical Design

The controller is a pure/testable state machine with n_cur, n_climb, and n_drop. Adaptive and fixed modes may vary speculative work but must preserve target semantics under deterministic sampling. Constants from the source are experimental hypotheses; offline replay and fixed-depth controls prevent overfitting. Aggregate TPS cannot conceal lower acceptance or changed token semantics.

## Code Samples & Guidance



## Files

patches/1255_nro06_adaptive_mtp_depth; adaptive controller/runtime policy; config and CLI; exhaustive unit tests; deterministic parity and balanced multi-content E2E evidence

## Validation

Exhaustive transition tests over floor/cap/acceptance/reset edges; multi-sequence independence; malformed configuration; deterministic output parity vs target/fixed MTP; acceptance trace integrity; long requests and request-boundary resets; balanced adaptive versus best-fixed controls over heterogeneous content and contexts.

## Effort & Risk



## Standards

Backend-neutral policy; source constants are hypotheses; no outcome-conditioned pair deletion; final-token correctness and work accounting required.

## Acceptance Criteria

All consolidated RD62/NRO06 requirements are explicit: state machine, reset boundaries, heterogeneous qualification, deterministic correctness, fixed-depth controls, instrumentation, and anti-overfitting. Fixed draft-mtp behavior remains unchanged; adaptive promotion requires improved results versus the best relevant fixed control across the pre-registered workload mix.

## Notes

Supersedes: RD62
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd62

Supersedes: RD62; consolidates duplicate NRO06 scope.
Inherited semantic scope: carry forward RD62's detailed state machine and NRO06's richer design/qualification constraints; historical source evidence remains on completed predecessors.
Migration: capability-rebaseline-v3-2026-09

## Change Log

- 2026-09-09T10:57:05.575180+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:14:19.351820+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.363199+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.151518+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:10:18.602333+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes
- chg_20260910_021313_the-semantic-audit-is-now-trac_4827
- 2026-09-10T02:13:13.335804+00:00 (updated-by): Updated: section:ledger-events
