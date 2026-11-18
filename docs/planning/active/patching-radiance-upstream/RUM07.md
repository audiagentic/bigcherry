---
id: RUM07
order: 7
plan: patching-radiance-upstream
state: pending
created-at: '2026-11-18T01:28:34.138312+00:00'
breadth: ''
skill: advanced
created-by: codex
priority: P1
work: L
---

# Consolidate KV layouts on upstream dtype row sizing and retain only supported local formats

## Description

Compare local bf16_i8 with upstream I8 numerics, reuse matching kernels and keep justified hybrid formats in one descriptor table. Port rad0037/38/40/41 and required selftests without restoring removed structs or duplicating layout switches.

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

Selected store/load/attention, stride and causal split-merge checks plus a short model quality canary in RUM09.

## Effort & Risk



## Standards

Shared agent workflow; tooling ownership and evidence boundaries. No deep testing during plan authoring.

## Acceptance Criteria

One row-size/capability source feeds planning and dispatch; retained formats pass supported geometries and unsupported shapes refuse explicitly.

## Notes

Imported stable roadmap identity from the reviewed design. Prerequisites: RUM03, RUM06. Review: docs/reviews/radiance-upstream-2026-11-18/README.md; proposal: MANIFEST-AND-EXTENSIONS.md. This initial item is the first-pass execution skeleton; detailed code guidance follows after the plan merge.

## Change Log

- 2026-11-18T01:28:34.138312+00:00 (created-by): Created by codex
