---
id: RCS20
order: 0
plan: build-radiance-components
state: completed
created-at: '2026-11-25T00:40:47.022098+00:00'
breadth: ''
skill: basic
created-by: lead
work: S
priority: P2
---

# Merge the kernel-family compile targets and record one-family against full-plugin build time

## Description

An earlier branch groups every upstream unit into eight families (core, gemm_quant, gemm_dense, attention, moe, speculation, multimodal, collectives) with compile-only per-family targets and a coverage test that each unit has exactly one family. It was never merged. It is the cheapest dev-loop gain and the starting point for loadable families.

## Steps

#1545 merged the eight family targets and coverage test. Use those actual Ninja object targets for developer and affected compile checks, not a second compile graph. Measure one representative family versus full plugin cold/warm during next actual builds; record dependency fan-out from Ninja/compiler output. Full plugin link/component tests only at existing integrated gate. Empty or excluded family is not qualification. No extra split purely to create families.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files

engines/radiance/components/libr3 (CMakeLists, family groups file, KERNEL_GROUPS.md, coverage test)

## Validation

Every upstream unit is covered by exactly one family; the full plugin links and the component tests pass; a one-family compile reuses the objects of the full build.

## Effort & Risk

Small; the branch is compile-only. One Ninja command per build directory.

## Standards



## Acceptance Criteria

The family targets are on main with a recorded one-family against full build time, cold and warm, and a family dependency-closure report.

## Notes

Design: docs/design/radiance-kernel-composition.md section 7.

2026-11-25 cross-programme bump review: docs/design/radiance-bump-integration.md is the shared implementation-order and compatibility specification; MEN24 owns sequence and timing, domain implementation remains here. No duplicate checker, bump driver, manifest launch or legacy fallback. Existing #1545 family compile targets and subsequent in-flight work are the single compile target owner. Expose affected-family builds to RCS19 and MEN16, actual same CMake flags/arch/object graph, reuse full-lane objects only with exact identity; changed shared header or unknown dependency widens compile set. No separate build system or incomplete family treated as production plugin.

2026-11-25 lean build sweep at 79c61a758. One existing CMake/Ninja build graph, one canonical matched release lane, existing tests and release records. No new cache, build system, manifest, hash scheme or generic framework. Reuse actual evidence; only test changed behavior and dependency closure. #1545 (0a03d2851) is on main. CMake family targets depend on canonical libr3 objects; no partial link or separate flags. Keep the coverage check; check actual requested unit coverage in partial builds, and use full target if shared inputs affect multiple families. Existing KERNEL_GROUPS.md and r3_groups_test.py suffice. Do not repeat delivered merge/build work; close only after remaining timing evidence.

2026-11-25 closed: family targets merged (#1545); one-family build time and dependency closure recorded in docs/evidence/2026-11-25-rcs20-rcs25-family-harness/README.md (IQR family 155 s with the selftest cached, 225 s cold, against a 25 to 28 min cached lane build; 29 project headers in the family closure, r3_compat.h and r3_mma16.h shared by every unit).

## Change Log

- 2026-11-25T00:40:47.022098+00:00 (created-by): Created by lead
- 2026-11-25T00:41:27.469152+00:00 (updated-by): Updated: section:description, section:steps, section:files, section:validation, section:effort_risk, section:acceptance_criteria, section:notes
- 2026-11-25T00:52:40.010572+00:00 (updated-by): Updated: section:steps, section:acceptance_criteria

## Reviews

- RV4606
- 2026-11-25T02:04:43.391147+00:00 (updated-by): Updated: section:notes
- 2026-11-25T05:21:03.549626+00:00 (updated-by): Updated: section:steps, section:notes
- 2026-11-25T08:45:07.808287+00:00 (updated-by): Updated: section:notes
- 2026-11-25T08:45:26.085769+00:00 (state-transition): State: pending → completed
