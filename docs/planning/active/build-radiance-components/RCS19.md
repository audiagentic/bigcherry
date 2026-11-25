---
id: RCS19
order: 0
plan: build-radiance-components
state: pending
created-at: '2026-11-25T00:40:42.311181+00:00'
breadth: ''
skill: intermediate
created-by: lead
work: S
priority: P1
---

# Kernel composition gates: provenance and anchors, compile, link and export, qualification; source ownership records and patched-source freshness

## Description

Managed patches and native rewrites both transform the same upstream kernel text. The gfx1201 fused-tile patch passed every gate and still could not build because its edits and the merged gfx1100 rewrites collided (anchors vanished, a type was defined twice); only a full lane build found it. Make every kernel change a unit with one owner per line of source, checked by a machine in seconds.

## Steps

1. Gate 1 (seconds, offline): apply the ordered patch set (bc patches included) to the pinned source, run the real rewrite check against it; exactly-one-of-pair for alternative groups; declared dependencies; region overlap; digest of the resulting source. 2. Gate 2: compile the affected translation units for gfx1100 and gfx1201 (host and device passes, architecture macros, explicit template instantiations) with failing fixtures: duplicate type, repeated template default argument, device-only definition clash, conflicting conditional compilation. 3. Gate 3: link the affected family or a contract fixture and check duplicate definitions, unresolved symbols, exported names, external device symbols. 4. Gate 4: resource, byte identity, selftest and replay for the affected family and the units that include a changed shared header. 5. Ownership records in patch metadata: affected units (expanded through the include graph), owned regions, consumed anchors. 6. Naming: architecture suffix only where two architectures must both define a name; uniqueness proven by gate 3; upstream and ABI names never renamed. 7. Freshness: make the libr3 generation depend on a digest of the patched inputs, with a test that changing an upstream header regenerates output without deleting the build directory. 8. First customer: the gfx1201 fused tile through the standard lane.

## Detailed Solution & Technical Design

Design: docs/design/radiance-kernel-composition.md section 4 (S1 to S5).

## Code Samples & Guidance



## Files

tools/bigcherry/release (patch checks), tools/tests/patch, engines/radiance/patches, skills/bigcherry-patch-author, docs/design/radiance-kernel-composition.md

## Validation

Fixtures: a vanished anchor, a doubled anchor and an exactly-one-of-pair violation fail gate 1; a duplicate type, a repeated template default and a device-only definition clash fail gate 2 and not gate 1; every existing patch set passes; the regeneration test passes; both architectures lane-build with the gfx1201 package applied and the gfx1100 resource records unchanged against the shipped release.

## Effort & Risk

Small. Risk: the gate needs the bc patches applied first to avoid false failures; keep the allowed-failure list explicit and tested.

## Standards



## Acceptance Criteria

Every managed patch set applies to the pinned source, every required anchor resolves under its declared variant, and every transformation has ownership and dependency metadata. Compile and link checks reject invalid declarations, duplicate definitions and symbol conflicts for both architectures before a lane build. No family is qualified until the affected and dependent kernels pass their correctness and resource gates.

## Notes

Stage 0 of the native RDNA3 migration brief (additive gfx12 body, distinct identifiers) is the same work.

2026-11-25 first customer: the gfx1201 fused-tile package. Resolved on a branch by keeping the patch's shared-header edits, making the five libr3 header blocks find-1-or-0 with a paired identity block (the rewriter's paired-run rule keeps upstream drift failing), renaming the patch's own tile to the gfx12-suffixed names and wrapping the tile and its dispatch so the gfx11 build ignores them. Lane build on both architectures: gfx1100 records unchanged against the shipped release (0 added, 0 removed, 0 changed), gfx1201 adds 11 fused kernels with 0 spills and the 44 plane variants unchanged. The anchor gate (step 1) would have caught this before the lane build.

## Change Log

- 2026-11-25T00:40:42.311181+00:00 (created-by): Created by lead
- 2026-11-25T00:41:22.694997+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:effort_risk, section:acceptance_criteria, section:notes
- 2026-11-25T00:42:02.400302+00:00 (updated-by): Updated: section:notes
- 2026-11-25T00:52:35.276328+00:00 (updated-by): Updated: section:title, section:steps, section:validation, section:acceptance_criteria
