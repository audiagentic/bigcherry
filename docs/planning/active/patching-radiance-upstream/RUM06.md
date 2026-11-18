---
id: RUM06
order: 6
plan: patching-radiance-upstream
state: pending
created-at: '2026-11-18T01:28:27.514165+00:00'
breadth: ''
skill: advanced
created-by: codex
priority: P0
work: L
---

# Port engine-scoped remote MoE and mixed-TP collective lifetimes

## Description

Port remote callbacks, parent tickets, layer maps and cached policy to engine/TP-group lifetime; reconcile per-group rendezvous, timeout rollback and chained fences. Preserve mixed TP ownership and all issue-ahead/cancel/drain rules; initially refuse unsupported multi-model helper operation.

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

Targeted lifecycle/admission/group tests followed by one representative mixed-layout canary at the acceptance gate.

## Effort & Risk



## Standards

Shared agent workflow; tooling ownership and evidence boundaries. No deep testing during plan authoring.

## Acceptance Criteria

No cross-group state or callback after free; selected production layouts and failure paths preserve correct ownership and rank progress.

## Notes

Imported stable roadmap identity from the reviewed design. Prerequisites: RUM02, RUM03, RUM05. Review: docs/reviews/radiance-upstream-2026-11-18/README.md; proposal: MANIFEST-AND-EXTENSIONS.md. This initial item is the first-pass execution skeleton; detailed code guidance follows after the plan merge.

## Change Log

- 2026-11-18T01:28:27.514165+00:00 (created-by): Created by codex
