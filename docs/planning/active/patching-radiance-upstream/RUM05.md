---
id: RUM05
order: 5
plan: patching-radiance-upstream
state: pending
created-at: '2026-11-18T01:28:20.204242+00:00'
breadth: ''
skill: advanced
created-by: codex
priority: P1
work: L
---

# Unify placement plans and helper reservations with upstream resource accounting

## Description

Implement the target-version portion through RHP13: one immutable owner/local-slot/encoding plan, local placement views and external helper reservations in upstream accounting. Reuse RHP04/RHP09 calibration; do not implement a second allocator or a new optimizer for the initial bump.

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

Focused owner-map, byte accounting, infeasible-plan and nonallocating planning tests.

## Effort & Risk



## Standards

Shared agent workflow; tooling ownership and evidence boundaries. No deep testing during plan authoring.

## Acceptance Criteria

Planner/runtime/helpers share one plan; queue/staging/KV/host/helper bytes are neither omitted nor double counted.

## Notes

Imported stable roadmap identity from the reviewed design. Prerequisites: RUM02, RUM03, RUM04. Review: docs/reviews/radiance-upstream-2026-11-18/README.md; proposal: MANIFEST-AND-EXTENSIONS.md. This initial item is the first-pass execution skeleton; detailed code guidance follows after the plan merge.

## Change Log

- 2026-11-18T01:28:20.204242+00:00 (created-by): Created by codex
