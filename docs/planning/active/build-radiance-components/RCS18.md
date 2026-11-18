---
id: RCS18
order: 0
plan: build-radiance-components
state: pending
created-at: '2026-11-18T08:41:57.683317+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# Switch supported Radiance releases to libr3d and retire the rewrite tree

## Description



## Steps

After RCS16 and RCS17 and matched workload gates, switch native build target, library declarations, build and runtime identities, SDK/package inventory, configs and launch callers together. Keep previous qualified release for rollback. Remove generated libr4d rewrite machinery and duplicate settings only after dependency and evidence audit.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Fresh release assemble/install, exact asset membership and ABI fingerprints, controlled selected-row trace and declared serving canary against frozen baseline; previous-release restore and loss-of-row refusal. No two implicit providers in the production release.

## Effort & Risk



## Standards



## Acceptance Criteria

Qualified releases select libr3d deliberately, all shipped paths have explicit disposition, and retired rewrite outputs are excluded from normal build/package/launch.

## Notes

Execution detail and source matrix: docs/reviews/radiance-native-libr3d/ASSESSMENT.md. Do not infer support from patch applicability or an upstream row name.

The surviving gfx1030 libr2 helper imports r3_rewrite.py, r3_compat.h, r3_gen_stubs.py and r3_cuid.cmake. Move its required subset to that existing owner and verify the gfx1030 helper build and drift checks before deleting the gfx1100 rewrite tree. Do not conflate this with adopting gfx11 native rows.

## Change Log

- 2026-11-18T08:41:57.683317+00:00 (created-by): Created by agent
- 2026-11-18T08:52:07.309399+00:00 (updated-by): Updated: section:notes
