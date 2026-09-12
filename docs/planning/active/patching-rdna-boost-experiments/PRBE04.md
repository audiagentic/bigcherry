---
id: PRBE04
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:53:43.516065+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: null
---

# Evaluate Q6_K MMQ sub-scale fold/hoist

## Description

Qualify the gfx1201 Q6_K MMQ sub-scale fold/hoist from patch 1203, preserving the existing BigCherry 1000/HI71 eligibility and PEF01 safety barriers.

## Steps

- Audit the exact Q6_K symbol/hunk against patch 1000 and resolve the current baseline including 1000.
- Run PEF01-specific safety, illegal-memory and correctness gates before timing.
- Compare baseline plus PRBE04 only on exact Q6_K shapes, with gfx1201 primary and gfx1100 control.
- Verify dense-shape-aware eligibility from HI71 and retain optional diagnostic arms only as explanatory.
- Record exact patch identity, shape matrix, fallback/quarantine decisions and balanced performance evidence.

## Detailed Solution & Technical Design

The production treatment is baseline plus PRBE04 where baseline includes validated 1000. Keep stock/1000 diagnostic arms explanatory only; do not bypass EX02/PEF01 or generalize eligibility beyond the proven Q6_K pattern.

## Code Samples & Guidance



## Files

patches/1203_rd050607_rdna4_wmma_fa_q6k_mmq; MMQ vec-dot/mmq sources; Q6_K fixtures; PEF01 quarantine evidence; HI71 eligibility cross-reference; campaign artifacts.

## Validation

Exact Q6_K shapes; gfx1201 primary; gfx1100 non-regression; output and memory safety; PEF01 reproduction/quarantine checks; HI71 dense-shape eligibility; balanced repeated performance.

## Effort & Risk



## Standards

PEF01 quarantine mandatory; exact identity; fail-closed promotion; never relax EX02.

## Acceptance Criteria

No illegal-memory or correctness failure occurs under PEF01/EX02 gates; only eligible Q6_K shapes select the treatment; gfx1201 evidence meets the registered performance boundary and gfx1100 is non-regressed, otherwise quarantine/reject.

## Notes

Supersedes: RD07
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd07

2026-09-11: started a real gfx1201 build+bench campaign for patch 1203 (same in-flight run as PRBE02/PRBE03). CAVEAT, important for this item specifically: this initial pass does NOT include PEF01's specific illegal-memory/safety gates or HI71's dense-shape eligibility verification, both of which this item's acceptance criteria mark as mandatory before any timing claim. Do not treat a clean PPL/bench result from this run as satisfying PRBE04 -- the PEF01 quarantine check and HI71 eligibility check are separate, not-yet-done, and higher-priority than the timing number given this item's own 'PEF01 quarantine mandatory... never relax EX02' standard.

REAL RESULT 2026-09-12: correctness (PPL-equality, see PRBE02) PASS, sigma=1.35. Performance (pp2048 +6.2%, pp512 +3.4%, likely partly attributable to RD07's Q6_K mmq fold specifically, though this run doesn't isolate RD05/RD06/RD07's individual contributions -- 1203 is one bundled patch). STILL BLOCKING per this item's own mandatory standard ('PEF01 quarantine mandatory... never relax EX02'): PEF01's illegal-memory/safety gate and HI71's dense-shape eligibility check were NOT run. Do not treat the real performance numbers above as satisfying this item -- they are encouraging first evidence, not a substitute for the mandatory safety gate.

## Change Log

- 2026-09-09T10:53:43.516065+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:10:22.973058+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.142855+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.816199+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:30:25.570783+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria
- chg_20260910_023037_three-rdna-boost-successors-no_6965
- 2026-09-10T02:30:37.095824+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-11T23:50:33.415840+00:00 (updated-by): Updated: section:notes
- chg_20260911_235103_caught-myself-running-a-real-h_5912
- 2026-09-11T23:51:03.386145+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-12T03:21:09.440786+00:00 (updated-by): Updated: section:notes
- chg_20260912_032330_ran-the-first-ever-real-hardwa_3337
- 2026-09-12T03:23:30.822091+00:00 (updated-by): Updated: section:ledger-events
