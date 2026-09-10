---
id: PRBE16
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:54:33.728988+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# Defer gfx1151 MMVQ nwarps=2 Q8_0 decode

## Description

Retain gfx1151 MMVQ nwarps=2 Q8_0 decode as a gate-verified-blocked hardware-specific experiment; do not claim Brutus validation.

## Steps

- Require PRBE19/RD25 prerequisite and preserve source 1818c3b... identity.
- Keep the exact gfx1151 guard and Q8_0 decode scope; obtain gfx1151 hardware before timing.
- On gfx1151 compare native control/treatment for exact shapes with correctness, resource/nwarps and decode evidence.
- Run explicit non-selection controls on gfx1030/gfx1100/gfx1201 using activation eligibility evidence.
- If hardware is unavailable, retain gate-verified-blocked disposition; do not extrapolate from current Brutus.

## Detailed Solution & Technical Design

Capability owner: patching

Split assessment: One independent boundary; Build/Run support is a dependency.

Overlap assessment: No duplicate boundary found; related items are prerequisites or adjacent evidence.

## Code Samples & Guidance



## Files

gfx1151 MMVQ catalog/source; activation eligibility evidence 1208_rd21_gfx1151_mmvq_nwarps_table; PRBE19 prerequisite; non-selection/build tests; gfx1151 campaign artifacts.

## Validation

Eligibility positive/negative architecture evidence; gfx1151 exact Q8_0 shapes; correctness; nwarps/resource; native comparison; no selection elsewhere.

## Effort & Risk



## Standards

Hardware-specific guard; no unsupported extrapolation; preserve gate-verified-blocked status.

## Acceptance Criteria

No current-hardware acceptance claim. Promotion requires gfx1151 evidence and exact guard/non-selection; until then the machine-checked blocked disposition remains authoritative.

## Notes

Supersedes: RD21
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd21

## Change Log

- 2026-09-09T10:54:33.728988+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:11:45.259387+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.201523+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.909397+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:49:12.755157+00:00 (updated-by): Updated: section:description, section:steps, section:files, section:validation, section:standards, section:acceptance_criteria
- chg_20260910_024925_rdna-successors-prbe1416-now_9529
- 2026-09-10T02:49:25.532036+00:00 (updated-by): Updated: section:ledger-events
