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

# Kernel composition seams: anchor gate, unit ownership and naming rules for patches and rewrites

## Description

Managed patches and native rewrites both transform the same upstream kernel text. The gfx1201 fused-tile patch passed every gate and still could not build because its edits and the merged gfx1100 rewrites collided (anchors vanished, a type was defined twice); only a full lane build found it. Make every kernel change a unit with one owner per line of source, checked by a machine in seconds.

## Steps

1. Anchor gate: apply the patch set to the pinned source, then run the rewrite check with the real rules against the patched source (allowed failures: the known bc_* sources, after applying the bc patches); put it in the offline patch job with a fixture reproducing the fused-tile collision. 2. Unit ownership: patch metadata declares the upstream units it edits; a check lists any unit edited by more than one layer. 3. Naming rule: a type, kernel or constant introduced by a patch or rewrite carries an architecture/patch suffix; lint rejects an unsuffixed introduction. 4. Layer order and the additive-first rule documented in the patch-author guidance. 5. First customer: the gfx1201 fused-tile package, rebuilt with distinct names and a separate gfx12 body, lane-built on both architectures, then promoted.

## Detailed Solution & Technical Design

Design: docs/design/radiance-kernel-composition.md section 4 (S1 to S5).

## Code Samples & Guidance



## Files

tools/bigcherry/release (patch checks), tools/tests/patch, engines/radiance/patches, skills/bigcherry-patch-author, docs/design/radiance-kernel-composition.md

## Validation

The collision that blocked the gfx1201 patch fails the gate in seconds on the unmodified package and passes on the fixed one; existing patches all pass the gate; patch tests and lint pass; both architectures lane-build with the gfx1201 package applied, the gfx1100 resource records unchanged against the shipped release.

## Effort & Risk

Small. Risk: the gate needs the bc patches applied first to avoid false failures; keep the allowed-failure list explicit and tested.

## Standards



## Acceptance Criteria

A patch or rewrite that edits another owner's anchored text, or defines an already-defined name, is refused before any lane build; the gfx1201 fused tile ships through the standard lane.

## Notes

Stage 0 of the native RDNA3 migration brief (additive gfx12 body, distinct identifiers) is the same work.

2026-11-25 first customer: the gfx1201 fused-tile package. Resolved on a branch by keeping the patch's shared-header edits, making the five libr3 header blocks find-1-or-0 with a paired identity block (the rewriter's paired-run rule keeps upstream drift failing), renaming the patch's own tile to the gfx12-suffixed names and wrapping the tile and its dispatch so the gfx11 build ignores them. Lane build on both architectures: gfx1100 records unchanged against the shipped release (0 added, 0 removed, 0 changed), gfx1201 adds 11 fused kernels with 0 spills and the 44 plane variants unchanged. The anchor gate (step 1) would have caught this before the lane build.

## Change Log

- 2026-11-25T00:40:42.311181+00:00 (created-by): Created by lead
- 2026-11-25T00:41:22.694997+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:effort_risk, section:acceptance_criteria, section:notes
- 2026-11-25T00:42:02.400302+00:00 (updated-by): Updated: section:notes
