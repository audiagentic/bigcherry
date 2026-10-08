---
id: PNRO20
order: 0
plan: patching-nasone-rdna-optimizations
state: completed
created-at: '2026-09-27T13:59:50.927188+00:00'
breadth: ''
skill: ''
created-by: agent
priority: P1
work: S
---

# 1253 test-backend-ops anchor fails when composed under 1205's evidence stack (blocks 1205 revalidation)

## Description

t-1205-gfx1201-s4 failed: once 1253_nro04_gfx1100_bf16_chunked_gdn was promoted into validated-enhancements, it is composed into every validation control/subject. Applied on top of 1205_rd12_paired_mmvq_dual_output's composition (which carries the test-backend-ops evidence patches), 1253's edit 'nro04-tests-01' on tests/test-backend-ops.cpp matches 0 times (anchor at the `double err = ud->tc->err(f1.data(), f2.data(), f1.size()); if (err > ud->tc->max_err(ud->backend1)) {` site). Another patch in the composed set rewrites that region first. Every future revalidation that composes both will fail closed the same way. Fix: re-anchor nro04-tests-01 to survive the evidence-stack rewrite (or order/declare the interaction explicitly), add a composition test covering 1253 + the test-backend-ops evidence patches, then requeue t-1205-gfx1201-s4.

## Steps



## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

The one-shot composition reproducer passed on local and Brutus trees; it was retired after closure and remains available in Git history.

## Effort & Risk



## Standards



## Acceptance Criteria

1222, 1223, 1253 and 1258 apply cleanly in composition order to the pinned test-backend-ops.cpp (repro_compose.py all OK); t-1205-gfx1201-s4 requeued.

## Notes

Found 2026-09-27 during a queue audit; the 1205 session had been sitting failed since before the lane split. Not a runtime bug -- patch-mechanics conflict only.

2026-09-28 RESOLVED (no code change needed): the failure was stale. t-1205-gfx1201-s4 failed at 2026-09-26 17:42:54; commit 5992f183 (2026-09-26 17:49:04, six minutes later) reworked 1223's anchor to insert after the pass/fail block so 1253's test hunks compose with it. Verified with the now-retired one-shot composition reproducer, which applies 1222 -> 1223 -> 1253 -> 1258 in composition order to test-backend-ops.cpp: all OK on both local and Brutus trees. Requeued t-1205-gfx1201-s4 (failed run moved aside as *.failed-pre5992f183).

## Change Log

- 2026-09-27T13:59:50.927188+00:00 (created-by): Created by agent
- 2026-09-27T14:06:12.153188+00:00 (updated-by): Updated: section:validation, section:acceptance_criteria, section:notes
- 2026-09-27T14:06:19.400684+00:00 (state-transition): State: pending → completed

## Ledger-events
