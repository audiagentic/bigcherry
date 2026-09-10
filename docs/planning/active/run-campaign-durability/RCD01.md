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

Keep generalized cross-process/host durable campaign restart dormant until real routine failures justify it; retain only the minimum rehydration protocol scope.

## Steps

Do not preemptively build. On reactivation, define typed campaign intent/run, complete operation specs, dependency-output bindings, persisted success descriptors and hash validation; rehydrate from empty process state and resume without rerunning completed stages; exclude distributed locks/host reclamation and generic workflow engine.

## Detailed Solution & Technical Design

HI82 already provides campaign-specific workdir identity/artifact validation/resume. Residual scope is reusable-build scheduler-grade restart across process/host interruption, triggered by evidence of routine multi-hour failures.

## Code Samples & Guidance



## Files

Reusable campaign run/intent schema, ArtifactStore rehydration integration, resume validator/tests and interruption fixtures when activated.

## Validation

Deferred until activation: persisted intent and stage outputs reproduce after process/host interruption without re-running completed stages.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Remain dormant until real campaign failures justify it; when activated, resume is hash-validated, deterministic, and minimum-protocol only.

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
- 2026-09-10T03:38:05.874609+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria
- chg_20260910_033817_repaired-vulkan-provenance-and_3027
- 2026-09-10T03:38:17.204821+00:00 (updated-by): Updated: section:ledger-events
