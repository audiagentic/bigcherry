---
id: RUM09
order: 9
plan: patching-radiance-upstream
state: pending
created-at: '2026-11-18T01:28:48.655352+00:00'
breadth: ''
skill: advanced
created-by: codex
priority: P0
work: M
---

# Prove the minimum working transition and publish the migrated runtime

## Description

Run only the matrix needed for the changed production configuration: composition/build/SDK, launch/stop/admission, helper-off control and representative helper-on mixed TP, short quality and paired performance checks. Tie evidence to exact source/build/data identities. Switch pin and publish after gates pass; retain previous verified release as recovery artifact, not a legacy code path.

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

Use existing acceptance thresholds; escalate testing only for an actual failure, unexplained drift or changed mode.

## Effort & Risk



## Standards

Shared agent workflow; tooling ownership and evidence boundaries. No deep testing during plan authoring.

## Acceptance Criteria

Required current production paths pass documented numerical/resource/lifecycle gates with no material unexplained regression; release bundle installs and runtime home contains one ABI generation.

## Notes

Imported stable roadmap identity from the reviewed design. Prerequisites: RUM03, RUM04, RUM05, RUM06, RUM07, RUM08. Review: docs/reviews/radiance-upstream-2026-11-18/README.md; proposal: MANIFEST-AND-EXTENSIONS.md. This initial item is the first-pass execution skeleton; detailed code guidance follows after the plan merge.

## Change Log

- 2026-11-18T01:28:48.655352+00:00 (created-by): Created by codex
