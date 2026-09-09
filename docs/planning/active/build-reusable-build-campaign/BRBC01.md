---
id: BRBC01
order: 0
plan: build-reusable-build-campaign
state: pending
created-at: '2026-09-09T10:59:07.257799+00:00'
breadth: ''
skill: intermediate
created-by: capability-rebaseline-v3
work: S
priority: null
---

# Restore an independent build-comparison oracle for future toolchain/RDNA-target changes

## Description

Evaluated against predecessor RE34 and the current build/campaign tree. No real toolchain, recipe, or RDNA-target change currently triggers this dormant oracle. The existing campaign comparison and check_parity() remain the supported path; creating a permanent second planner now would violate the predecessor scope.

## Steps

1. Implement the still-valid future scope.
2. Run the stated acceptance and evidence gates.
3. Preserve predecessor provenance and record successor evidence under this ID.

## Detailed Solution & Technical Design

Capability owner: build

Split assessment: One independent boundary; Build/Run support is a dependency.

Overlap assessment: No duplicate boundary found; related items are prerequisites or adjacent evidence.

## Code Samples & Guidance



## Files

successor-specs/build-reusable-build-campaign-re34.md

## Validation

No implementation is authorized until a concrete trigger exists. At trigger time, follow RE34's independent-construction recipe: freeze campaign BuildPlan A, construct B through a separate documented CMake/toolchain path, publish both through ArtifactStore, run check_parity plus explicit option/source/toolchain diff, and add the shared-omission negative test.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Close the recorded gap and pass the frozen scope's stated implementation, correctness, performance, or evidence gate.

## Notes

Supersedes: RE34
Migration: capability-rebaseline-v3-2026-09
Successor key: build-reusable-build-campaign-re34

2026-09-10 evaluation: intentionally remains pending/dormant; no trigger exists. Do not mark completed merely because the current campaign path is healthy, and do not implement speculative duplicate build machinery.

## Change Log

- 2026-09-09T10:59:07.257799+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:16:16.339662+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.484239+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-09T15:27:41.040607+00:00 (updated-by): Updated: section:description, section:validation, section:notes
- chg_20260909_152750_confirmed-brbc01-is-deliberate_9064
- 2026-09-09T15:27:50.763518+00:00 (updated-by): Updated: section:ledger-events
