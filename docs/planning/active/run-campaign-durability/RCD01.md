---
id: RCD01
order: 0
plan: run-campaign-durability
state: pending
created-at: '2026-09-09T10:47:10.667188+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: null
---

# Durable campaign restart/resume (deferred from reusable-build-campaign RE11)

## Description

Deferred RE11 durability protocol remains explicitly retained; cross-process run resume is unfinished.

## Steps

1. Implement the still-valid future scope.
2. Run the stated acceptance and evidence gates.
3. Preserve predecessor provenance and record successor evidence under this ID.

## Detailed Solution & Technical Design

Capability owner: run

Split assessment: One independent boundary; Build/Run support is a dependency.

Overlap assessment: No duplicate boundary found; related items are prerequisites or adjacent evidence.

## Code Samples & Guidance



## Files

tools/bigcherry/campaign_graph.py, tools/bigcherry/campaign.py, tools/bigcherry/resources.py (per RE11's original Files list -- verify these still exist under current tools/bigcherry/campaign/ layout before reactivating; project has reorganized since RE11 was filed 2026-08-15).

## Validation

DORMANT -- not evaluated this pass. When reactivated: NEGATIVE/FALSIFICATION per RE11 -- kill the process mid-run, then resume; a tampered intent is rejected; output artifacts are re-byte-verified before reuse; a stage left 'running' by the crash does not surface as success; two concurrent processes cannot corrupt each other's run state; changing any field of the operation spec changes stage identity.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Close the recorded gap and pass the frozen scope's stated implementation, correctness, performance, or evidence gate.

## Notes

Supersedes: CD01
Migration: capability-rebaseline-v3-2026-09
Successor key: run-campaign-durability-cd01

External dev-gpt holistic review (2026-09-10, req_9f60aaa2ae5f4b88): CONFIRMED PARK / do not implement now. Predecessor design is sufficient as a dormant reference; add one explicit activation trigger not previously stated: reactivate only after repeated REAL campaign failures that actually require cross-process/host recovery (not preemptively). Before any eventual implementation, resolve cross-host fencing/ownership (how a new host proves the former executor cannot publish while cross-host lock-breaking remains forbidden). Preserve the RE11 design's operation-spec vs execution-hash split, immutable result records, byte verification, and interrupted-never-equals-success rule. Execution order: explicitly NOT ranked in the 1-12 sequence — stays dormant.

RE-ASSESSED 2026-09-10 against the WHOLE project run history (per user directive), independently of the earlier narrower review. Verdict: CONFIRMED KEEP DORMANT, activation trigger still not met. Evidence checked: repo history contains many real crashes/failures (including tonight's RU01/PHC03 META segfault, and known RCCL abort paths), but these are workload/runtime CORRECTNESS failures, not campaign-STATE durability failures requiring cross-process/host recovery -- a different failure class than this item's own trigger. HI82 already demonstrates successful controlled STOP/resume for at least one real campaign. Closest real incident found: one pin-bump run crashed and left a stale marker, recovered via existing manual reset+resume -- a single incident, not the 'repeated real campaign failures' this item's own activation trigger explicitly requires. No repeated evidence anywhere in commit history, the release ledger, or planning/evidence history of lost/ambiguous campaign state, unsafe resume, or repeated manual reconstruction after process/host loss. Repo evidence is sufficient to decide NOT to activate now -- cross-host fencing/ownership design details would only need to be filled in if/when activation eventually happens. No action taken; correctly remains a dormant, activation-gated reference item, not actionable run work.

## Change Log

- 2026-09-09T10:47:10.667188+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:03:16.931644+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:00.731253+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T00:07:59.576114+00:00 (updated-by): Updated: section:notes
- chg_20260910_000828_reviewed-and-re-planned-all-pe_3612
- 2026-09-10T00:08:28.898595+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.162086+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T00:27:35.811237+00:00 (updated-by): Updated: section:files, section:validation
- 2026-09-10T02:55:52.402689+00:00 (updated-by): Updated: section:notes
- chg_20260910_025604_confirmed-via-a-fresh-whole-pr_7836
- 2026-09-10T02:56:04.074592+00:00 (updated-by): Updated: section:ledger-events
