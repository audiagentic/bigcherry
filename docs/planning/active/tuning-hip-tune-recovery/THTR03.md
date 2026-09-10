---
id: THTR03
order: 0
plan: tuning-hip-tune-recovery
state: pending
created-at: '2026-09-09T10:51:29.984447+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# Empirical multiplicity-correction investigation (BH vs Holm vs BY) -- study first, no production change

## Description

Empirically compare BH, BY, and Holm multiplicity correction under BigCherry's real measurement dependence before changing production policy.

## Steps

Use archived native-twin/known-null campaign data preserving cross-signature correlation; compare false-discovery behavior under BH/BY/Holm; keep fixed prospective hypothesis family; only if material BH risk appears design versioned MultiplicityPolicy, otherwise document no change.

## Detailed Solution & Technical Design

HI141 behavioral failure is orthogonal to performance FDR and does not justify changing BH. Correction method may become pluggable; family definition must never be runtime-configurable.

## Code Samples & Guidance



## Files

Offline study tooling/report and archived campaign artifacts; future policy interface only if warranted.

## Validation

Empirical FDR/power comparison using real correlation, not only independent uniform simulation.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

No production correction change without material empirical risk; any future policy is versioned while hypothesis-family boundaries stay fixed.

## Notes

Supersedes: HTR05
Migration: capability-rebaseline-v3-2026-09
Successor key: tuning-hip-tune-recovery-htr05

## Change Log

- 2026-09-09T10:51:29.984447+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:07:53.235105+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.021353+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.633220+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:36:39.858389+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria
- chg_20260910_033709_repaired-five-recovery-success_3695
- 2026-09-10T03:37:09.543479+00:00 (updated-by): Updated: section:ledger-events
