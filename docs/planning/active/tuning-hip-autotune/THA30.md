---
id: THA30
order: 0
plan: tuning-hip-autotune
state: pending
created-at: '2026-09-09T10:51:03.277738+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: null
---

# Replay fallback paths under graph capture (hardware separation, cache corruption, process-cache hit)

## Description

Test replay fallback under graph capture across hardware separation, missing/corrupt/incompatible caches, process cache hits, and captured launch correctness.

## Steps

Construct wrong-hardware, missing-candidate, corrupted, incompatible-manifest caches; verify native/refined fallback; run graph capture with replay and confirm captured launch is resolved candidate, not repeated lookup; verify process cache avoids hashing after first lookup.

## Detailed Solution & Technical Design

Split from HI16; preserve already-closed catalog/digest/forced-native scope. Fallback must be fail-closed and graph capture must retain the selected launch.

## Code Samples & Guidance



## Files

Replay/cache loader, graph capture tests, cache fixtures and optional hardware validation.

## Validation

Real test for every fallback and graph/process-cache scenario, ideally hardware validated.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

All fallback conditions select safe native/refined behavior, graph capture records correct launch, and warm process cache eliminates recurring lookup work.

## Notes

Supersedes: HI94
Migration: capability-rebaseline-v3-2026-09
Successor key: tuning-hip-autotune-hi94

## Change Log

- 2026-09-09T10:51:03.277738+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:07:23.231983+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:00.992747+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.592706+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:35:05.268462+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria
- chg_20260910_033527_repaired-five-more-tuning-succ_3062
- 2026-09-10T03:35:27.089190+00:00 (updated-by): Updated: section:ledger-events
