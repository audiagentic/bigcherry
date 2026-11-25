---
id: RCS20
order: 0
plan: build-radiance-components
state: pending
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

1. Refresh the branch against main. 2. Full libr3 build and the component tests. 3. Record cold and warm build time of one family against the full plugin, and report the include-graph closure per family (a changed shared header rebuilds which families). 4. Merge. The family targets are a development accelerator; they are not evidence that separate family libraries link (that is RCS22).

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

## Change Log

- 2026-11-25T00:40:47.022098+00:00 (created-by): Created by lead
- 2026-11-25T00:41:27.469152+00:00 (updated-by): Updated: section:description, section:steps, section:files, section:validation, section:effort_risk, section:acceptance_criteria, section:notes
- 2026-11-25T00:52:40.010572+00:00 (updated-by): Updated: section:steps, section:acceptance_criteria

## Reviews

- RV4606
- 2026-11-25T02:04:43.391147+00:00 (updated-by): Updated: section:notes
