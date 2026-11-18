---
id: RUM04
order: 4
plan: patching-radiance-upstream
state: pending
created-at: '2026-11-18T01:28:13.721371+00:00'
breadth: ''
skill: advanced
created-by: codex
priority: P0
work: L
---

# Reuse native fleet YAML and compile typed BigCherry intent into an immutable runtime bundle

## Description

Implement native settings through upstream YAML validation with closed top-level x-bigcherry intent. Keep generated per-model contracts, digest/provenance and live admission. Migrate current configuration callers together; coordinate RSC12 rather than create a competing resolver.

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

Focused YAML parity, strict extension decoding, digest/admission, launch and reload identity tests.

## Effort & Risk



## Standards

Shared agent workflow; tooling ownership and evidence boundaries. No deep testing during plan authoring.

## Acceptance Criteria

One authored YAML source resolves once; helpers and engine consume the same admitted generation; no argv bridge or independently editable duplicate settings remain.

## Notes

Imported stable roadmap identity from the reviewed design. Prerequisites: RUM01, RUM02, RUM03. Review: docs/reviews/radiance-upstream-2026-11-18/README.md; proposal: MANIFEST-AND-EXTENSIONS.md. This initial item is the first-pass execution skeleton; detailed code guidance follows after the plan merge.

## Change Log

- 2026-11-18T01:28:13.721371+00:00 (created-by): Created by codex
