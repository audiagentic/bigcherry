---
id: RUM10
order: 10
plan: patching-radiance-upstream
state: pending
created-at: '2026-11-18T01:28:59.863720+00:00'
breadth: ''
skill: advanced
created-by: codex
priority: P3
work: L
---

# Add multi-model fleet helper support only after the single-model migration

## Description

Optional later enhancement, not a prerequisite for the upstream bump. Add fleet-wide helper ownership, per-card power conflict handling, park/reload/reclaim and group-id reuse after stable single-model parity.

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

Small two-model lifecycle/resource-isolation cases, adding hardware contention checks only for activated shared-card behavior.

## Effort & Risk



## Standards

Shared agent workflow; tooling ownership and evidence boundaries. No deep testing during plan authoring.

## Acceptance Criteria

Multiple models cannot share stale helper state or overcommit reservations; unsupported modes remain explicitly refused until this item is qualified.

## Notes

Imported stable roadmap identity from the reviewed design. Prerequisites: RUM09. Review: docs/reviews/radiance-upstream-2026-11-18/README.md; proposal: MANIFEST-AND-EXTENSIONS.md. This initial item is the first-pass execution skeleton; detailed code guidance follows after the plan merge.

## Change Log

- 2026-11-18T01:28:59.863720+00:00 (created-by): Created by codex
