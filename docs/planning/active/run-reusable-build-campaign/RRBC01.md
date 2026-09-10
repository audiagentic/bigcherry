---
id: RRBC01
order: 5
plan: run-reusable-build-campaign
state: pending
created-at: '2026-09-09T10:58:54.395708+00:00'
breadth: ''
skill: intermediate
created-by: capability-rebaseline-v3
work: S
priority: P2
---

# Vulkan-aware device-visibility wiring for build's runtime smoke path

## Description

The frozen reusable-build-campaign item remains active with implementation/acceptance work outstanding and no terminal disposition.

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

successor-specs/run-reusable-build-campaign-re31.md

## Validation

Historical evidence and constraints: Preserve frozen notes/reviews/evidence on predecessor; IDs: none extracted.

Active dependencies: Frozen dependencies: RE30.

Reference handling: Rewrite forward references (2); preserve historical references (0) on predecessor.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Close the recorded gap and pass the frozen scope's stated implementation, correctness, performance, or evidence gate.

## Notes

Supersedes: RE31
Migration: capability-rebaseline-v3-2026-09
Successor key: run-reusable-build-campaign-re31

External dev-gpt holistic review (2026-09-10, req_9f60aaa2ae5f4b88): GO, design sufficient as-is. CampaignLane.backend is the authoritative source of which visibility variable to emit; caller supplies a backend-neutral visibility selection and the adapter emits exactly one backend visibility env var. Reject conflicting inherited HIP/Vulkan visibility state rather than guessing which one wins. Re-derive current Vulkan<->HIP device-enumeration-order mapping fresh on real hardware before acceptance (do not trust the stale RE30-era mapping). Execution order: ranked #5 (removes the first concrete HIP-only assumption from build smoke; prerequisite for RRBC02/RRVP01 lane work).

## Change Log

- 2026-09-09T10:58:54.395708+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:16:03.776505+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.471692+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T00:07:36.484700+00:00 (updated-by): Updated: section:notes
- 2026-09-10T00:08:05.332242+00:00 (updated-by): Updated: order=5, priority='P2'
- chg_20260910_000828_reviewed-and-re-planned-all-pe_3612
- 2026-09-10T00:08:28.943223+00:00 (updated-by): Updated: section:ledger-events
