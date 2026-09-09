---
id: BRBE01
order: 0
plan: build-rdna-boost-experiments
state: completed
created-at: '2026-09-09T10:53:55.572615+00:00'
breadth: ''
skill: intermediate
created-by: capability-rebaseline-v3
work: S
priority: null
---

# RESOLVED: run_id path leak + lock ownership bug (both fixed); original instant-fake-ok symptom unexplained (low priority)

## Description

Terminal successor for the resolved RD100 investigation. The remaining low-priority instant-success symptom did not reproduce after the lock-ownership and stable-generated-input fixes; the actionable defects are fixed and the campaign build/record/tune/promote path has real cross-architecture validation. No additional implementation remains in this successor.

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

successor-specs/build-rdna-boost-experiments-rd100.md

## Validation

Inherited historical evidence remains on RD100 and is not reassigned: commits 2a88de9, b54b26a, and 1623637 fixed lock ownership, stage-error visibility, and run-id-dependent generated-input identity; RD100 records real Brutus validation across gfx1100/gfx1201, dense/MoE models, and multiple quantizations. The original instant-success symptom remained non-reproducible after those fixes and is explicitly retained as historical low-priority uncertainty, not an active acceptance gap. Successor acceptance is therefore satisfied by the verified fixes and cross-architecture campaign validation already recorded by its predecessor.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

- The actionable run_id/lock/build-reuse defects are fixed and have regression coverage.
- The fixed build -> record -> tune -> promote path has real validation across more than one GPU architecture/model shape.
- The unexplained historical instant-success symptom is not reclassified as solved; it remains documented on RD100 as non-reproduced low-priority history.
- No further implementation is owned by BRBE01.

## Notes

Supersedes: RD100
Migration: capability-rebaseline-v3-2026-09
Successor key: build-rdna-boost-experiments-rd100

Supersedes: RD100
Migration: capability-rebaseline-v3-2026-09
Successor key: build-rdna-boost-experiments-rd100

Terminal disposition: completed. Historical evidence remains attributed to RD100; this successor closes only the still-valid continuation after confirming no active gap remains.

## Change Log

- 2026-09-09T10:53:55.572615+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:11:01.046850+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.156209+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-09T14:53:46.946653+00:00 (updated-by): Updated: section:description, section:validation, section:acceptance_criteria, section:notes
- chg_20260909_145357_closed-brbe01-after-confirming_3643
- 2026-09-09T14:53:57.076475+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-09T14:54:06.444462+00:00 (state-transition): State: pending → completed
