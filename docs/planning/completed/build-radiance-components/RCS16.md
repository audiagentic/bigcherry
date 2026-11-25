---
id: RCS16
order: 0
plan: build-radiance-components
state: deprecated
created-at: '2026-11-18T08:41:37.901602+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: L
---

# Retain qualified native RDNA3 rows; own required compute deltas as overlays

## Description



## Steps

After RCS15 native-provider decision by actual row/format, use stock libr3d where measured support/equivalence exists; do not modify upstream repository or backport candidate rows to old pin. Missing or optimized local compute rows are owned overlays under RCS23 and RCS21 traits, using native plugin interface. RCS16 owns inventory/retention and exact gap requirements, RCS23 implementation of overlay families; only narrow managed native source patches for necessary interface/semantic seams, not duplicate compute implementation. Begin supported int8 and packed IQR source comparison, then local u6/u7, group256 and special attention/KV and FlashNext forms. Retain qualified local base where native lacks closure or regresses. RCS17/18 own mixed integration and conditional release retirement.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

One native reference comparison and shape-refusal case per changed path, same fixture row selection and matched timing. Gate formats and replay on RUM07/RUM08. Stop on layout, quality or resource mismatch.

## Effort & Risk



## Standards



## Acceptance Criteria

Selected gfx1100 model paths run with native rows plus explicit minimal deltas. No hidden stock/libref fallback and no wholesale generated rewrite.

Every required model path has actual selected native or explicitly owned overlay row, tested layout, schema, shape/format and numerical/perf evidence; unsupported native routes retain qualified provider with explicit disposition. No unapproved CPU/reference fallback, no whole-library fork or duplicate implementation under RCS16/RCS23.

## Notes

Execution detail and source matrix: docs/reviews/radiance-native-libr3d/ASSESSMENT.md. Do not infer support from patch applicability or an upstream row name.

At b8bca788 native h128 attention does not cover local Flash-Next h256/GQA6, and the actual Q27 INT8 fixture uses i8a8h rather than plain w8a8. Inspect selected rows, matrix operand packing, KV bytes and reachable reference gaps before designing each port. Preserve local higher-M and split-store behavior only where required.

2026-11-25 second integration review: docs/design/radiance-integration-review-20261125.md is the exact-source review and controls conflicting earlier wording. Shipping pin remains b6b8751 (2.2.0), upstream observed main 1c104de4 is a development candidate, 119 commits and 327 changed files; no automatic repin or backport. No new compatibility layer, launch manifest, bump driver or identity machinery. Original instruction to port every delta into upstream libr3d is superseded by overlay-first design #1553. New upstream native int8/IQR rows are adoption candidates, not replacements by name; preserve existing local wins and independent RUM15 precision issue.

2026-11-25 closed as not adopted. The RCS15 baseline (docs/evidence/2026-11-25-rcs15-libr3d-inventory and the IQR row in RCS15) found no upstream native gfx1100 row that beats ours where we own the work: upstream's IQR GEMM is scalar and 850 to 1650 times slower at 129 and 2048 rows. Our qualified rows stay; no overlay migration is needed to keep them (RCS23 closed the same way).

## Change Log

- 2026-11-18T08:41:37.901602+00:00 (created-by): Created by agent
- 2026-11-18T08:49:08.981231+00:00 (updated-by): Updated: section:notes

## Reviews

- RV4628
- 2026-11-25T02:38:20.324015+00:00 (updated-by): Updated: section:title, section:steps, section:notes, section:acceptance_criteria
- 2026-11-25T09:42:10.454262+00:00 (updated-by): Updated: section:notes
- 2026-11-25T09:42:46.436986+00:00 (state-transition): State: pending → deprecated
