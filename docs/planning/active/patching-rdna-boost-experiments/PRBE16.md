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

TODO, but gate-verified-blocked: gfx1151 hardware is not present on Brutus, so this item cannot progress past eligibility-gate verification until gfx1151 hardware exists in this project's fleet. Patch 1208_rd21_gfx1151_mmvq_nwarps_table already exists (state=untested) and defines the gfx1151 MMVQ nwarps=2 Q8_0 decode eligibility table/guard.

## Steps

1. Verify patches/1208_rd21_gfx1151_mmvq_nwarps_table/patch.toml and patch.py still define the exact gfx1151 guard and Q8_0 decode scope described in this item (re-read the current patch.py Edit() anchors before any other step -- do not assume they are unchanged).
2. Run patch-lint and patch-rebase-check on 1208 against the current bigcherry-tuning pin -- this is real, runnable work even without gfx1151 hardware.
3. Run explicit non-selection controls on the hardware this project DOES have (gfx1030/gfx1100/gfx1201): confirm 1208's calc_nwarps()/eligibility table does NOT select the gfx1151 nwarps=2 path on these architectures (build + activation-trace check, no gfx1151-specific timing implied).
4. Record the current disposition as gate-verified-blocked: non-selection controls pass on available hardware, but no gfx1151 correctness/decode/timing evidence exists and none should be claimed.
5. Do not extrapolate current Brutus (gfx1030/gfx1100/gfx1201) results to gfx1151 -- leave the item pending/blocked rather than promoting on partial evidence.

## Detailed Solution & Technical Design

This item's own acceptance criteria already state no current-hardware acceptance claim is possible. The only real, honest work available right now is (a) offline lint/rebase-check on patch 1208, (b) non-selection proof on the three architectures this project actually has. Both are concrete, runnable today. Full promotion remains blocked on gfx1151 hardware acquisition, which is outside this item's control.

## Code Samples & Guidance



## Files

patches/1208_rd21_gfx1151_mmvq_nwarps_table/{patch.toml,patch.py}; gfx1030/gfx1100/gfx1201 non-selection test fixtures.

## Validation

Offline: `PYTHONPATH=tools python -m bigcherry patch-lint`, `patch-rebase-check --focal-overlay 1208_rd21_gfx1151_mmvq_nwarps_table --source bigcherry-tuning`. Hardware (Brutus, available architectures only): build + activation-trace non-selection check on gfx1030/gfx1100/gfx1201 confirming 1208's table never selects the gfx1151 nwarps=2 Q8_0 path there. No gfx1151 evidence is obtainable until that hardware exists in the fleet -- explicitly leave that gap open rather than closing this item.

## Effort & Risk

S effort for the available non-selection/offline work; full closure is blocked on hardware acquisition (not an engineering risk, an availability constraint).

## Standards

Hardware-specific guard; no unsupported extrapolation; preserve gate-verified-blocked status.

## Acceptance Criteria

No current-hardware acceptance claim. Promotion requires gfx1151 evidence and exact guard/non-selection; until then the machine-checked blocked disposition remains authoritative.

## Notes

Supersedes: RD21
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd21

Supersedes: RD21 (closed historical predecessor). Preserve source 1818c3b... as provenance. PRBE19 is a constraint applied while materializing calc_nwarps(), not a live prerequisite patch.

2026-09-24 relevance at b11126: TODO/gate-verified-blocked, unchanged disposition -- patch 1208 exists and is untested; real progress is limited to offline lint/rebase-check plus non-selection controls on available (non-gfx1151) hardware. No GPT design request used -- this item has no open kernel-design question, only an availability blocker and procedural non-selection verification.

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
- 2026-09-12T09:52:24.101421+00:00 (updated-by): Updated: section:steps, section:files, section:notes
- chg_20260912_095506_cleaned-the-active-rdna-boost_4906
- 2026-09-12T09:55:06.232507+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T02:29:09.948585+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:effort_risk, section:notes
