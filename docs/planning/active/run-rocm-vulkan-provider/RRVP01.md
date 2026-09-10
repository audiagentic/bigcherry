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

Thread explicit stack_name through CampaignRequest, lane/execution specs, planner backend checks, CLI/workflow stages, receipts, and lane identity while preserving legacy --lane syntax.

## Steps

Add stack selection/absence semantics; parse source:build:platform plus stack internally; thread through build/record/tune/profile/verifier/replay and receipt reconstruction; reject missing/ambiguous/unknown/backend-mismatched stack before evidence execution; add legacy compatibility and collision tests.

## Detailed Solution & Technical Design

Selected stack is execution intent and campaign identity, distinct from runtime attestation. Resolve cfg.stacks once, preserve public three-part lane grammar, and make source:build:platform:stack internal identity.

## Code Samples & Guidance



## Files

campaign planner/lane/CLI build/tuning/profiling/workflow and tests.

## Validation

All stack stages preserve stack_name in receipts; legacy lanes parse; distinct stacks cannot collide; mismatch fails closed.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Stack-aware campaign identity is threaded end-to-end, legacy syntax remains compatible, and invalid or mismatched stacks fail before evidence-producing work.

## Notes

Supersedes: RO02
Migration: capability-rebaseline-v3-2026-09
Successor key: run-rocm-vulkan-provider-ro02

External dev-gpt holistic review (2026-09-10, req_9f60aaa2ae5f4b88): GO once absence semantics are made explicit. --lane may keep its 3-part external CLI syntax, but resolved INTERNAL lane identity must always include stack. Stack selection comes from an explicit request field or an explicit configured default — never inferred from PATH/env or dynamically "pick what exists." Unknown/missing/ambiguous/backend-mismatched stack must hard-fail before any work starts (RV126's prior confirmation that RO01 stays authoritative and this stays tooling/workflow-only, not a second stack-selection path, still holds). Execution order: ranked #6 — establish canonical stack selection through campaign intent/receipts before anything downstream hashes or probes it.

STRONGLY CONFIRMED via deeper repo-validated dev-gpt review (2026-09-10): checked CampaignRequest, CampaignLane, CampaignLaneSelector, and lane_id() directly -- none currently contain any stack selection/identity. RO01's BackendStack config already exists and is unused downstream, so this item is exactly the missing threading layer, not speculative work. Execution order shifts to #5 in the revised sequence.

PAUSED 2026-09-10 (user directive): Vulkan is out of scope for now -- plans may continue to be updated/reviewed, but implementation is paused. This item's design remains as reviewed above (GO once absence semantics explicit); a partial implementation was started (CampaignLane.stack_name, CampaignRequest.stack, _resolve_stack() fail-closed validation, lane_id() including stack) and then REVERTED uncommitted rather than landed, specifically because making stack mandatory on every plan() call has a real blast radius across cli/build.py, profiling/workflow.py, tuning/workflow.py, and two test files that needs its own deliberate pass -- not something to rush through under a paused-scope directive. Do not resume implementation until Vulkan work is unpaused; the design/order above stays valid for when it is.

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
- 2026-09-10T00:25:54.282915+00:00 (updated-by): Updated: section:notes
- chg_20260910_002605_paused-all-vulkan-provider-imp_6846
- 2026-09-10T00:26:05.819568+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T00:27:47.677492+00:00 (updated-by): Updated: section:files, section:validation
- 2026-09-10T03:28:45.445536+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria
- chg_20260910_032911_repaired-three-providerrun-su_5934
- 2026-09-10T03:29:11.969018+00:00 (updated-by): Updated: section:ledger-events
