---
id: THA31
order: 0
plan: tuning-hip-autotune
state: pending
created-at: '2026-09-09T10:51:07.086823+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# Adaptive retention band derived from the screen-stage canary (HI24 step 7)

## Description

Derive screen-stage retention band from same-kernel canary under the settled no-flush execution contract.

## Steps

Compute canary_screen_pct before retention; derive band=clamp(2×canary, screen_keep_within_pct, band_max_pct); use dispersion fallback when no pair; validate quiet/noisy runs, identical preconditioning, no flush widening, and THA27 no-flush baseline.

## Detailed Solution & Technical Design

Quiet runs floor at 10%; RV21-style ~14% canary widens near 28%; never exceed max. This changes candidate admission, so require real hardware and regression comparison.

## Code Samples & Guidance



## Files

Screen-stage canary/retention logic, HI24 contract tests, real hardware evidence.

## Validation

Quiet finalist sets byte-identical to current; noisy retention widens appropriately; cap enforced; real RX7900 GRE validation.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Band is reproducibly derived, bounded, and improves noisy-run retention without changing quiet behavior or violating no-flush policy.

## Notes

Supersedes: HI95
Migration: capability-rebaseline-v3-2026-09
Successor key: tuning-hip-autotune-hi95

## Change Log

- 2026-09-09T10:51:07.086823+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:07:27.875647+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:00.998833+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.599655+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:35:13.824060+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria
- chg_20260910_033527_repaired-five-more-tuning-succ_3062
- 2026-09-10T03:35:27.102590+00:00 (updated-by): Updated: section:ledger-events
