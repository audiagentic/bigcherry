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

Keep generalized cross-process/host durable campaign restart dormant until repeated real campaign-state failures justify it, while preserving the complete minimum hash-validated rehydration protocol as the activation boundary.

## Steps

Do not preemptively build. On activation, implement typed CampaignIntent/Run and complete OperationSpec records with deterministic graph ordering, GPU/build/source claims, dependency-output bindings, and immutable stage evidence. Persist request/materialize/build intent before work; persist typed running/result states with operation_spec_hash, execution_hash, input/output artifact IDs and output content hashes. Resume only after rehydrating and byte/descriptor-verifying dependency outputs and exact result identity; restore completed stages without rerunning them. Treat running without result as interrupted, never success; reject request/intent tampering, operation-field changes, dependency-output hash changes, and concurrent state corruption. Restore materialize source_slice_id, and permit only explicit recovery of a dead local owner; never auto-break another-host/unknown lock. Resolve cross-host fencing/ownership before any host reclamation design; exclude distributed lock reclamation and a generic workflow engine.

## Detailed Solution & Technical Design

HI82 already provides campaign-specific workdir identity/artifact validation/resume. The residual scheduler-grade protocol is a durable static intent plus complete operation spec, dependency-output binding, immutable result record, and rehydration across process/host interruption. Static operation_spec_hash captures every behavior-affecting operation field; execution_hash additionally binds dependency outputs, so unchanged build specs still re-execute when generated inputs differ. Stages publish running.json as a crash marker, then atomically publish immutable result.json only after output rehydration/verification. Resume restores ArtifactRefs and materialize source_slice_id, blocks descendants on failures, and never equates interrupted with success.

## Code Samples & Guidance



## Files

CampaignIntent/OperationSpec/StageRecord schemas; campaign graph/executor/resource ownership; ArtifactStore rehydration and resume validator; interruption/tamper/concurrency fixtures; activation evidence and cross-host fencing design.

## Validation

Activation is allowed only after repeated real campaign failures requiring cross-process/host recovery, not ordinary workload/runtime correctness failures. Then run falsification tests: kill after running marker; tamper request/intent; alter every operation field (environment, architecture, binary path, n_gen); change dependency output with same static spec; byte-flip output; tamper descriptors; restore downstream generate/build outputs; verify running≠success; exercise concurrent processes and dead local-owner recovery; prove another-host/unknown locks are never auto-broken and materialize source_slice_id is restored.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Remain dormant until the explicit repeated-campaign-state-failure trigger is met. When activated, the minimum protocol preserves typed plans/runs, deterministic graph order, GPU/build/source claims, immutable stage evidence, operation-versus-execution hash separation, byte/descriptor re-verification, interrupted-state safety, tamper rejection, dependency invalidation, concurrency safety, materialize restoration, and constrained local-lock recovery. No distributed lock reclamation or generic workflow engine is added.

## Notes

Supersedes: CD01
Migration: capability-rebaseline-v3-2026-09
Successor key: run-campaign-durability-cd01

External dev-gpt holistic review (2026-09-10, req_9f60aaa2ae5f4b88): CONFIRMED PARK / do not implement now. Predecessor design is sufficient as a dormant reference; add one explicit activation trigger not previously stated: reactivate only after repeated REAL campaign failures that actually require cross-process/host recovery (not preemptively). Before any eventual implementation, resolve cross-host fencing/ownership (how a new host proves the former executor cannot publish while cross-host lock-breaking remains forbidden). Preserve the RE11 design's operation-spec vs execution-hash split, immutable result records, byte verification, and interrupted-never-equals-success rule. Execution order: explicitly NOT ranked in the 1-12 sequence — stays dormant.

RE-ASSESSED 2026-09-10 against the WHOLE project run history (per user directive), independently of the earlier narrower review. Verdict: CONFIRMED KEEP DORMANT, activation trigger still not met. Evidence checked: repo history contains many real crashes/failures (including tonight's RU01/PHC03 META segfault, and known RCCL abort paths), but these are workload/runtime CORRECTNESS failures, not campaign-STATE durability failures requiring cross-process/host recovery -- a different failure class than this item's own trigger. HI82 already demonstrates successful controlled STOP/resume for at least one real campaign. Closest real incident found: one pin-bump run crashed and left a stale marker, recovered via existing manual reset+resume -- a single incident, not the 'repeated real campaign failures' this item's own activation trigger explicitly requires. No repeated evidence anywhere in commit history, the release ledger, or planning/evidence history of lost/ambiguous campaign state, unsafe resume, or repeated manual reconstruction after process/host loss. Repo evidence is sufficient to decide NOT to activate now -- cross-host fencing/ownership design details would only need to be filled in if/when activation eventually happens. No action taken; correctly remains a dormant, activation-gated reference item, not actionable run work.

Supersedes CD01. Preserve RE11's durable-run design inline rather than relying on the deprecated predecessor: operation-spec versus execution-hash split, immutable results, byte verification, running≠success, all falsifiers, and cross-host fencing boundary. Existing HI82 resume does not satisfy this generalized scope.

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
- 2026-09-10T04:02:12.649207+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria, section:notes
- chg_20260910_040226_fixed-the-four-remaining-seman_2499
- 2026-09-10T04:02:26.621227+00:00 (updated-by): Updated: section:ledger-events
