---
id: RRVP02
order: 7
plan: run-rocm-vulkan-provider
state: pending
created-at: '2026-09-09T10:59:48.032974+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: P2
---

# Backend stack probing and canonical identity

## Description

Canonical backend identity/probing implementation is pending; acceptance is unchecked.

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

successor-specs/run-rocm-vulkan-provider-ro03.md

## Validation

Historical evidence and constraints: Preserve frozen notes/reviews/evidence on predecessor; IDs: none extracted.

Active dependencies: Frozen dependencies: RO02.

Reference handling: Rewrite forward references (4); preserve historical references (0) on predecessor.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Close the recorded gap and pass the frozen scope's stated implementation, correctness, performance, or evidence gate.

## Notes

Supersedes: RO03
Migration: capability-rebaseline-v3-2026-09
Successor key: run-rocm-vulkan-provider-ro03

External dev-gpt holistic review (2026-09-10, req_9f60aaa2ae5f4b88): FREEZE WIRE MODEL then GO. Conceptual design (RV127's HIP/Vulkan-neutral envelope) is correct; one residual mixing to fix: extensions/features/limits/device facts belong in CapabilitySnapshot, NOT in ResolvedStackIdentity — the latter is software/provider identity only. Freeze canonical component/module identity representation, canonical JSON encoding, version/domain tagging, and field-ordering rules BEFORE writing probes. One shared implementation/schema for both HIP and Vulkan. Confirmed sole authority for "expected resolved HIP/Vulkan software/provider identity" (ResolvedStackIdentity) and "device/runtime capabilities" (CapabilitySnapshot) — RRBC02 must consume these, never invent a parallel Vulkan identity digest (see RRBC02's own notes for the reciprocal resolution). Execution order: ranked #7.

## Change Log

- 2026-09-09T10:59:48.032974+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:16:51.254873+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.519983+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T00:07:41.712289+00:00 (updated-by): Updated: section:notes
- 2026-09-10T00:08:06.761608+00:00 (updated-by): Updated: order=7, priority='P2'
- chg_20260910_000828_reviewed-and-re-planned-all-pe_3612
- 2026-09-10T00:08:28.964266+00:00 (updated-by): Updated: section:ledger-events
