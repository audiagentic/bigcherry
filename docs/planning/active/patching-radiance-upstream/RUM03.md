---
id: RUM03
order: 3
plan: patching-radiance-upstream
state: pending
created-at: '2026-11-18T01:28:07.390588+00:00'
breadth: ''
skill: advanced
created-by: codex
priority: P0
work: M
---

# Port build dependencies and plugin ABI 41 with a consistent runtime SDK

## Description

Decouple BigCherry nlohmann dependency from removed upstream vendor header. Rebuild local plugins against one ABI/layout fingerprint; update Registry shared-set row/device eligibility without losing chain_fenced.

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

Configure/build relevant components and SDK; targeted plugin ABI mismatch and code-eligibility checks.

## Effort & Risk



## Standards

Shared agent workflow; tooling ownership and evidence boundaries. No deep testing during plan authoring.

## Acceptance Criteria

One coherent engine/plugin SDK builds, old binaries are refused and release inputs include every dependency.

## Notes

Imported stable roadmap identity from the reviewed design. Prerequisites: RUM01. Review: docs/reviews/radiance-upstream-2026-11-18/README.md; proposal: MANIFEST-AND-EXTENSIONS.md. This initial item is the first-pass execution skeleton; detailed code guidance follows after the plan merge.

## Change Log

- 2026-11-18T01:28:07.390588+00:00 (created-by): Created by codex
