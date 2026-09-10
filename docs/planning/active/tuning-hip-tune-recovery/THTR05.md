---
id: THTR05
order: 0
plan: tuning-hip-tune-recovery
state: pending
created-at: '2026-09-09T10:51:38.359111+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: P0
---

# Adaptive recovery: bisect with the shortest vector that actually reproduces the failure

## Description

Optimize recovery bisection by selecting the shortest vector that reproduces the same first acceptance divergence, with full-corpus final validation.

## Steps

Run full corpus once; identify failing vectors and first divergence; select shortest reproducer, optionally minimize n_predict; use only that oracle for bisection/alternative search; cache deterministic native trace; validate final candidate cache against immutable full corpus before publish.

## Detailed Solution & Technical Design

Never use a cheap vector that cannot observe the failure. Oracle selection is anchored to same first divergence, guarding against isolating a different defect.

## Code Samples & Guidance



## Files

Recovery oracle selection/bisection, native trace cache, full-corpus validator and tests.

## Validation

Known failure oracle reproduces same divergence, isolates same signature as full bisection, final cache passes full corpus.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Search cost is reduced without changing isolated cause or evidence quality; publication always requires full immutable corpus validation.

## Notes

Supersedes: HTR07
Migration: capability-rebaseline-v3-2026-09
Successor key: tuning-hip-tune-recovery-htr07

## Change Log

- 2026-09-09T10:51:38.359111+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:08:03.783654+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.030411+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.648480+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:36:54.896836+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria
- chg_20260910_033709_repaired-five-recovery-success_3695
- 2026-09-10T03:37:09.569044+00:00 (updated-by): Updated: section:ledger-events
