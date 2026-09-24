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

1. Verify patches/1208_rd21_gfx1151_mmvq_nwarps_table/patch.toml and patch.py still define the exact gfx1151 guard and Q8_0 decode scope described in this item -- CORRECTED per source audit: the patch currently contains NO BIGCHERRY_PATCH_TRACE/BIGCHERRY_PATCH_HIT instrumentation at all, so add host-side trace instrumentation before any non-selection check can be evidenced (see step 3).
2. Run patch-lint and patch-rebase-check on 1208 against the current bigcherry-tuning pin.
3. Add a BIGCHERRY_PATCH_TRACE-gated GGML_LOG_WARN marker to the RDNA3_5 selection/launch path -- CORRECTED: place it on the host-side selection/launch site (e.g. around get_device_table_id(int cc) / the MMVQ launch after Q8_0+ncols eligibility is known), NOT inside the constexpr __host__ __device__ calc_nwarps() body, since a device-callable constexpr function is the wrong instrumentation point.
4. Run explicit non-selection controls on gfx1030/gfx1100/gfx1201 using the new marker: confirm 1208's calc_nwarps()/eligibility table does NOT select the gfx1151 nwarps=2 path on these architectures. CORRECTED scope: validate the full documented ncols_dst range 1..8 (this item's own baked-in table deliberately selects nwarps=2 for ncols_dst <= MMVQ_MAX_BATCH_SIZE, i.e. 1..8), not only ncols=1.
5. Add the PRBE19 provenance/equivalence check against reviewed commit 9e46e1fd... and snapshot c8af5361... (currently absent from this item).
6. Record the current disposition as gate-verified-blocked: non-selection controls pass on available hardware across ncols 1..8, but no gfx1151 correctness/decode/timing evidence exists and none should be claimed.

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

2026-09-24 GPT review req_7f4dea253b7247f0 applied: patch 1208 has no BIGCHERRY_PATCH_TRACE/HIT instrumentation today -- added a required host-side trace marker (not inside the device-callable constexpr calc_nwarps body). Widened the non-selection validation scope from ncols=1 only to the full documented ncols_dst 1..8 range (nwarps=2 is baked in for ncols_dst <= MMVQ_MAX_BATCH_SIZE). Added the missing PRBE19 provenance/equivalence check (9e46e1fd... vs c8af5361...).

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
- 2026-09-24T04:39:45.501359+00:00 (updated-by): Updated: section:steps, section:notes
