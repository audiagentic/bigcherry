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

TODO, rescoped off dead identity. PRBE04's target (RD07 Q6_K MMQ sub-scale fold/hoist prefill work) originally lived inside patch 1203, now `rejected` as a bundle for RD06's failure (PA39). RD07 itself passed CLEANLY across gfx1100/gfx1201/gfx1030 within that run (activation, backend-reference, performance, controls all PASS) -- the strongest real result of the three RD05/06/07 slices -- but PA39's own decision (GPT lifecycle review req_f34f50a25c6240fe) voids reuse of 1203's receipt: RD07 may only return as a new, separately-identified `untested` patch with fresh hardware evidence. PRBE110 (separate plan item, already created) owns authoring that new package (rd07-* edits: mmq-vec-dot.cuh fold/hoist/sum-line, mmq.cu activation marker, J_MAX env, test-backend-ops perf cases). PRBE04 owns the downstream re-qualification of that new identity, including the PEF01/HI71 gates this item's own notes flag as never having actually been run under ANY identity.

## Steps

1. Confirm PRBE110 has landed the new RD07-only patch package (bound to RD07-Q6K-MMQ-PREFILL-FOLD contract per PRBE110's own step 1); if not landed, stay blocked rather than qualifying the dead 1203 identity.
2. Audit the exact Q6_K symbol/hunk in the new package against patch 1000/HI71 (dense-shape-aware eligibility) to resolve the current baseline composition including 1000.
3. Run PEF01's specific illegal-memory/safety gates FIRST -- this item's own notes explicitly flag that the 2026-09-12 real-hardware pass did NOT include these and that a clean timing number must not be treated as satisfying this item without them (this item's own standard: 'PEF01 quarantine mandatory... never relax EX02').
4. Run HI71 dense-shape eligibility verification to confirm the treatment only selects eligible Q6_K shapes, not a generalized pattern.
5. Compare baseline+new-RD07-patch only, on exact Q6_K shapes, gfx1201 primary with gfx1100 non-regression control -- repeat the same shape matrix and repetition discipline as the old (now-voided) 1203 run, but under the new patch identity and citing fresh evidence.
6. Record the new patch's resolved identity, PEF01/HI71 gate results, shape matrix, and fallback/quarantine decision; only then does the prior RD07 performance signal (pp2048 +6.2%, pp512 +3.4% under the old 1203 identity) become reusable as a directional expectation, not as closing evidence.

## Detailed Solution & Technical Design

This is the highest-confidence candidate of the three ex-1203 slices (it passed cleanly on all three architectures under the old identity) but is still blocked from promotion by two mandatory, never-yet-run gates: PEF01 illegal-memory/safety and HI71 dense-shape eligibility. The re-extraction under a new identity is an opportunity to close both gates properly before any timing claim, exactly as this item's pre-existing standards already require.

## Code Samples & Guidance

No new anchors proposed here -- PRBE110 authors the RD07-only patch.py (fold/hoist of Q6_K mmq sub-scales into the row base-scale, same family as patches/1204_rd08_q6k_mmvq_vdr2's vecdotq.cuh-style edits). PRBE04's deliverable is the PEF01/HI71 gate harness plus the qualification campaign against that package once it exists.

## Files

Depends on PRBE110's package name (TBD, e.g. patches/12xx_rd07_q6k_mmq_prefill_fold/); ggml/src/ggml-cuda/mmq.cu and vecdotq.cuh (Q6_K mmq sub-scale path); patch 1000/HI71 eligibility cross-reference; PEF01 illegal-memory/safety fixtures; campaign artifacts for the new identity.

## Validation

Offline: `PYTHONPATH=tools python -m bigcherry patch-lint`, `patch-rebase-check --focal-overlay <new-rd07-id> --source bigcherry-tuning` once PRBE110's package exists. Hardware (Brutus, not run here): PEF01 illegal-memory/safety gate FIRST; HI71 dense-shape eligibility check; exact Q6_K shapes on gfx1201 primary, gfx1100 non-regression control; balanced repeated llama-bench performance only after both safety gates pass.

## Effort & Risk

M effort (RD07's design is already proven correct and performant under the old identity; the real remaining work is the two mandatory safety/eligibility gates that were never run) -- blocked until PRBE110 delivers the new patch identity.

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

2026-09-24 relevance at b11126: TODO, blocked on PRBE110; PEF01 safety gate and HI71 eligibility check are the real, never-yet-closed gaps per this item's own prior notes, not the timing claim (which already passed cleanly under the old, now-voided identity). GPT design request submitted (req_990c48138f9b408e, batched with PRBE02); gateway was heavily congested at submission time -- if it doesn't complete, this plan was authored directly against real patch.toml/SUMMARY.md evidence and this item's own prior real-hardware notes.

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
- 2026-09-24T02:31:54.027638+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:effort_risk, section:notes
