---
id: RRBC02
order: 10
plan: run-reusable-build-campaign
state: pending
created-at: '2026-09-09T10:59:03.207047+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: P3
---

# Replace platform.vulkan-linux's targets placeholder with real device/ICD identity

## Description

Real Vulkan device and ICD identity replacement remains planned.

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

successor-specs/run-reusable-build-campaign-re33.md

## Validation

Historical evidence and constraints: Preserve frozen notes/reviews/evidence on predecessor; IDs: EC01,EC05,EC15.

Active dependencies: Frozen dependencies: RE30,RE31,RE32.

Reference handling: Rewrite forward references (4); preserve historical references (3) on predecessor.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Close the recorded gap and pass the frozen scope's stated implementation, correctness, performance, or evidence gate.

## Notes

Supersedes: RE33
Migration: capability-rebaseline-v3-2026-09
Successor key: run-reusable-build-campaign-re33

External dev-gpt holistic review (2026-09-10, req_9f60aaa2ae5f4b88): RE-SCOPE BEFORE CODING. The predecessor (RE33) content wrongly mixes declarative platform constraints, observed software identity, and device capabilities into one item. Re-scope this item to own ONLY Vulkan platform constraints/selection policy (vendor/device-class constraints, ICD selection policy, minimum Vulkan API/capability requirements) — it must consume RRVP02's identity/capability outputs, never invent a parallel Vulkan identity digest. New dependency added: RRBC02 -> RRVP02 (this item cannot start before RRVP02 is frozen). Remove the Vulkan `targets` config field entirely rather than retaining it as a compatibility placeholder (project doctrine: no legacy shims). RRBC02 references RRVP02 fingerprints/snapshots directly. Naming resolution: keep RRBC02 for the RE33 lineage and RRVP02 for the RO03 lineage — do not rename/merge one into the other, they are genuinely separate authorities (RRBC02 = declarative constraint/policy; RRVP02 = probed ResolvedStackIdentity + CapabilitySnapshot). Experiment corpora must retain an explicit ICD/driver identity partition per EC15's folded-in requirement — never pool across ICDs. Execution order: ranked #10, after RRVP02/BRVP01/RRVP03 (moved later than RE33's original position since it now depends on RRVP02).

## Change Log

- 2026-09-09T10:59:03.207047+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:16:11.720645+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events



- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.480385+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T00:07:51.075586+00:00 (updated-by): Updated: section:notes
- 2026-09-10T00:08:08.881022+00:00 (updated-by): Updated: order=10, priority='P3'
- chg_20260910_000828_reviewed-and-re-planned-all-pe_3612
- 2026-09-10T00:08:28.947678+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.333474+00:00 (updated-by): Updated: section:ledger-events
