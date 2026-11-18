---
id: RUM02
order: 2
plan: patching-radiance-upstream
state: pending
created-at: '2026-11-18T01:28:00.655993+00:00'
breadth: ''
skill: advanced
created-by: codex
priority: P1
work: L
---

# Extract injected BigCherry implementations into owned components and narrow hooks

## Description

Move large source-creation literals and duplicated policy from patch.py into ordinary owned source files. Keep only concrete typed integration hooks; reuse RCS13 callable inventory and RCS01 ownership mapping. Preserve behavior before mixing extraction with upstream ports.

## Steps

- Read the preserved upstream review and manifest/extension proposal.
- Implement the scoped change in a small PR; preserve unrelated work.
- Run only the targeted validation listed here; retain failed or unqualified paths as pending.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files

- docs/reviews/radiance-upstream-2026-11-18/README.md
- docs/reviews/radiance-upstream-2026-11-18/MANIFEST-AND-EXTENSIONS.md

## Validation

Focused extraction/build-identity and baseline behavior checks; no broad performance campaign.

## Effort & Risk



## Standards

Shared agent workflow; tooling ownership and evidence boundaries. No deep testing during plan authoring.

## Acceptance Criteria

Owned implementations compile through declared CMake inputs; upstream patch footprint is limited to necessary hooks and has no repeated configuration parsing in execution.

## Notes

Imported stable roadmap identity from the reviewed design. Prerequisites: RUM01. Review: docs/reviews/radiance-upstream-2026-11-18/README.md; proposal: MANIFEST-AND-EXTENSIONS.md. This initial item is the first-pass execution skeleton; detailed code guidance follows after the plan merge.

## Change Log

- 2026-11-18T01:28:00.655993+00:00 (created-by): Created by codex
