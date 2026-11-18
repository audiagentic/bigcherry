---
id: RUM08
order: 8
plan: patching-radiance-upstream
state: pending
created-at: '2026-11-18T01:28:40.856862+00:00'
breadth: ''
skill: advanced
created-by: codex
priority: P1
work: M
---

# Reuse upstream MTP replay and reconcile scheduling policy

## Description

Compare upstream recording against rad0013 before retirement; select one scheduler controller and preserve rank-consistent draft/history handling. Keep adaptive/traversal/suffix modes outside the initial acceptance matrix unless explicitly enabled.

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

Focused replay-key/rank-agreement/prefix-preemption checks and a fixed-depth production canary.

## Effort & Risk



## Standards

Shared agent workflow; tooling ownership and evidence boundaries. No deep testing during plan authoring.

## Acceptance Criteria

No redundant replay override or competing controller; accepted history and defaults are explicit for the initial production mode.

## Notes

Imported stable roadmap identity from the reviewed design. Prerequisites: RUM04, RUM06, RUM07. Review: docs/reviews/radiance-upstream-2026-11-18/README.md; proposal: MANIFEST-AND-EXTENSIONS.md. This initial item is the first-pass execution skeleton; detailed code guidance follows after the plan merge.

## Change Log

- 2026-11-18T01:28:40.856862+00:00 (created-by): Created by codex
