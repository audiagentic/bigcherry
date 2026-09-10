---
id: RRVP01
order: 5
plan: run-rocm-vulkan-provider
state: pending
created-at: '2026-09-09T10:59:42.884350+00:00'
breadth: ''
skill: intermediate
created-by: capability-rebaseline-v3
work: M
priority: P2
---

# Campaign and CLI stack propagation

## Description

Stack propagation work is pending; acceptance remains unchecked.

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

successor-specs/run-rocm-vulkan-provider-ro02.md

## Validation

Historical evidence and constraints: Preserve frozen notes/reviews/evidence on predecessor; IDs: RO01.

Active dependencies: Frozen dependencies: RO01.

Reference handling: Rewrite forward references (1); preserve historical references (1) on predecessor.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Close the recorded gap and pass the frozen scope's stated implementation, correctness, performance, or evidence gate.

## Notes

Supersedes: RO02
Migration: capability-rebaseline-v3-2026-09
Successor key: run-rocm-vulkan-provider-ro02

External dev-gpt holistic review (2026-09-10, req_9f60aaa2ae5f4b88): GO once absence semantics are made explicit. --lane may keep its 3-part external CLI syntax, but resolved INTERNAL lane identity must always include stack. Stack selection comes from an explicit request field or an explicit configured default — never inferred from PATH/env or dynamically "pick what exists." Unknown/missing/ambiguous/backend-mismatched stack must hard-fail before any work starts (RV126's prior confirmation that RO01 stays authoritative and this stays tooling/workflow-only, not a second stack-selection path, still holds). Execution order: ranked #6 — establish canonical stack selection through campaign intent/receipts before anything downstream hashes or probes it.

STRONGLY CONFIRMED via deeper repo-validated dev-gpt review (2026-09-10): checked CampaignRequest, CampaignLane, CampaignLaneSelector, and lane_id() directly -- none currently contain any stack selection/identity. RO01's BackendStack config already exists and is unused downstream, so this item is exactly the missing threading layer, not speculative work. Execution order shifts to #5 in the revised sequence.

## Change Log

- 2026-09-09T10:59:42.884350+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:16:46.946452+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.515849+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T00:07:38.803173+00:00 (updated-by): Updated: section:notes
- 2026-09-10T00:08:06.058689+00:00 (updated-by): Updated: order=6, priority='P2'
- chg_20260910_000828_reviewed-and-re-planned-all-pe_3612
- 2026-09-10T00:08:28.959982+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.393549+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T00:18:55.355103+00:00 (updated-by): Updated: order=5, section:notes
