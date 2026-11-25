---
id: RCS18
order: 0
plan: build-radiance-components
state: deprecated
created-at: '2026-11-18T08:41:57.683317+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# Conditionally adopt native libr3d and retire only proven-obsolete generated paths

## Description



## Steps

RCS15 coverage/performance and RCS16/23 local overlay closure plus RCS17 transport gate are prerequisites, not commitment to switch. Prefer native provider only for qualified supported paths; preserve generated provider if required rows or quality/perf remain incomplete. At actual release cutover update existing plugin hierarchy in YAML, build/SDK/package inventory and callers together; no silent fallback or compatibility shim. Remove generated gfx1100 machinery only when every active user is proven obsolete; gfx1030 libr2 shared rewrite/generator tools must move to existing owner with actual helper build and drift gate before deletion. Whole previous installed release is rollback.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Fresh release assemble/install, exact asset membership and ABI fingerprints, controlled selected-row trace and declared serving canary against frozen baseline; previous-release restore and loss-of-row refusal. No two implicit providers in the production release.

## Effort & Risk



## Standards



## Acceptance Criteria

Qualified releases select libr3d deliberately, all shipped paths have explicit disposition, and retired rewrite outputs are excluded from normal build/package/launch.

Deliberate per-layout/provider adoption backed by actual model, selected rows and complete quality/resource/performance/transport proof; generated outputs retired only if no production dependency (including gfx1030 tools). If native is not qualified, document retained provider and negative evidence, do not mark full retirement complete. Installed Gate E and affected mixed configs, previous release rollback, missing-provider refusal.

## Notes

Execution detail and source matrix: docs/reviews/radiance-native-libr3d/ASSESSMENT.md. Do not infer support from patch applicability or an upstream row name.

The surviving gfx1030 libr2 helper imports r3_rewrite.py, r3_compat.h, r3_gen_stubs.py and r3_cuid.cmake. Move its required subset to that existing owner and verify the gfx1030 helper build and drift checks before deleting the gfx1100 rewrite tree. Do not conflate this with adopting gfx11 native rows.

2026-11-25 second integration review: docs/design/radiance-integration-review-20261125.md is the exact-source review and controls conflicting earlier wording. Shipping pin remains b6b8751 (2.2.0), upstream observed main 1c104de4 is a development candidate, 119 commits and 327 changed files; no automatic repin or backport. No new compatibility layer, launch manifest, bump driver or identity machinery. Candidate native U3/U4/U5 IQR scalar row does not cover local u6/u7 or all IQ8R groups; CMake exact gfx1100 target condition. Do not confuse discovery with qualified release. No forced switch or duplicate library migration before RCS15 decision.

2026-11-25 closed as not adopted: no native libr3d row qualified (RCS15, RCS16). The generated paths are not retired; the libr3 rewrites, the family targets (#1545), the composition gates (#1552, #1580, #1584) and the unit harness (#1565) stand.

## Change Log

- 2026-11-18T08:41:57.683317+00:00 (created-by): Created by agent
- 2026-11-18T08:52:07.309399+00:00 (updated-by): Updated: section:notes

## Reviews

- RV4630
- 2026-11-25T02:39:22.226560+00:00 (updated-by): Updated: section:title, section:steps, section:notes, section:acceptance_criteria
- 2026-11-25T09:42:20.096402+00:00 (updated-by): Updated: section:notes
- 2026-11-25T09:42:56.124438+00:00 (state-transition): State: pending → deprecated
