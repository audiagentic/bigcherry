---
id: RUM01
order: 1
plan: patching-radiance-upstream
state: pending
created-at: '2026-11-18T01:27:53.871983+00:00'
breadth: ''
skill: advanced
created-by: codex
priority: P0
work: M
---

# Approve target identities and the minimal Radiance 2.x migration architecture

## Description

Decide the exact target commit, native YAML ownership, generated runtime-contract boundary, small extension interfaces and single-model-first scope. Link RSC12/RSC13, RCS01/RCS08/RCS14, RHP04/RHP09/RHP13 and PRQ qualification owners. No new allocator or generic plugin framework.

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

Review the pinned source interfaces and each local patch disposition; check dependency ordering, not hardware.

## Effort & Risk



## Standards

Shared agent workflow; tooling ownership and evidence boundaries. No deep testing during plan authoring.

## Acceptance Criteria

A recorded decision names one authoring authority, one allocator, owned hook boundaries and the minimum migration matrix.

## Notes

Imported stable roadmap identity from the reviewed design. Prerequisites: none. Review: docs/reviews/radiance-upstream-2026-11-18/README.md; proposal: MANIFEST-AND-EXTENSIONS.md. This initial item is the first-pass execution skeleton; detailed code guidance follows after the plan merge.

## Change Log

- 2026-11-18T01:27:53.871983+00:00 (created-by): Created by codex
