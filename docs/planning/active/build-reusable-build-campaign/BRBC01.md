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

Dormant conditional oracle from RE34. No current trigger exists; preserve this as a pending reference-only gate and do not build a second planner.

## Steps

- When a concrete toolchain, recipe or RDNA-target change needs an independent oracle, freeze BuildPlan A.
- Construct BuildPlan B through a separately documented CMake/toolchain path; publish both through ArtifactStore.
- Run check_parity plus explicit option/source/toolchain diffs and the shared-omission negative test.
- If no trigger exists, leave this item dormant and do not implement speculative duplicate campaign machinery.

## Detailed Solution & Technical Design

Capability owner: build

Split assessment: One independent boundary; Build/Run support is a dependency.

Overlap assessment: No duplicate boundary found; related items are prerequisites or adjacent evidence.

## Code Samples & Guidance



## Files

Existing campaign BuildPlan, ArtifactStore, check_parity() and independent construction recipe; future trigger record and comparison artifacts.

## Validation

No implementation before trigger. At trigger: independent A/B construction, artifact publication, check_parity, source/toolchain/options diff and shared-omission negative test.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Pending/dormant is the correct state while no concrete trigger exists. Once triggered, the independent oracle passes all listed comparison and negative gates; no second planner is introduced.

## Notes

Supersedes: RE34
Migration: capability-rebaseline-v3-2026-09
Successor key: build-reusable-build-campaign-re34

2026-09-10 evaluation: intentionally remains pending/dormant; no trigger exists. Do not mark completed merely because the current campaign path is healthy, and do not implement speculative duplicate build machinery.

Supersedes: RE34
semantic-carryforward: dispositioned dormant/no-trigger; concrete trigger remains required.

## Change Log

- 2026-09-09T10:59:07.257799+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:16:16.339662+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.484239+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-09T15:27:41.040607+00:00 (updated-by): Updated: section:description, section:validation, section:notes
- chg_20260909_152750_confirmed-brbc01-is-deliberate_9064
- 2026-09-09T15:27:50.763518+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.339381+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:40:57.120886+00:00 (updated-by): Updated: section:description, section:steps, section:files, section:validation, section:acceptance_criteria, section:notes
- chg_20260910_024129_build-and-external-fix-success_1105
- 2026-09-10T02:41:29.953651+00:00 (updated-by): Updated: section:ledger-events
