---
id: RRVP02
order: 6
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

PAUSED (Vulkan) -- when resumed: tools/bigcherry/backend/{__init__,stack,hip_probe,vulkan_probe}.py (new), plus renamed RuntimeCapabilitySnapshot per the naming-collision correction above.

## Validation

PAUSED (Vulkan). Probe twice -> byte-identical JSON/fingerprint; provider-binary swap -> fingerprint changes; device-visibility/ordinal reordering -> fingerprint UNCHANGED; missing/unknown/unsupported states round-trip explicitly (never inferred).

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

CORRECTION from deeper repo-validated dev-gpt review (2026-09-10): architecture (ResolvedStackIdentity vs CapabilitySnapshot split) remains right, but "CapabilitySnapshot" collides in name with this project's EXISTING tuning CapabilityMask128/HIP producer-capability system, which describes a different thing (producer semantic-compatibility bits, not runtime device/API/provider facts). Rename this item's concept to RuntimeCapabilitySnapshot (or similarly disambiguated) before implementing, to avoid two same-named-but-different capability concepts in the codebase. Execution order shifts to #6 in the revised sequence.

PAUSED 2026-09-10 (user directive): Vulkan is out of scope for now -- plans may continue to be updated/reviewed, implementation is paused. Design above (freeze wire model, rename to RuntimeCapabilitySnapshot) stands for when resumed.

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
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.400828+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T00:18:57.949636+00:00 (updated-by): Updated: order=6, section:notes
- 2026-09-10T00:25:55.832992+00:00 (updated-by): Updated: section:notes
- chg_20260910_002605_paused-all-vulkan-provider-imp_6846
- 2026-09-10T00:26:05.841918+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T00:27:50.126765+00:00 (updated-by): Updated: section:files, section:validation
