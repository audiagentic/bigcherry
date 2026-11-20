---
id: RCS16
order: 0
plan: build-radiance-components
state: pending
created-at: '2026-11-18T08:41:37.901602+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: L
---

# Move required RDNA3 compute deltas into upstream libr3d

## Description



## Steps

After RCS15, retain native rows where schema, geometry, storage and behavior match. Add only required gaps at upstream row and HIP unit seams, one family per patch. Start with dense kernels, then attention/KV and selected Flash-Next math. Prefer local patches against libr3d that can be proposed upstream; reuse its tuning axes and launch shims.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

One native reference comparison and shape-refusal case per changed path, same fixture row selection and matched timing. Gate formats and replay on RUM07/RUM08. Stop on layout, quality or resource mismatch.

## Effort & Risk



## Standards



## Acceptance Criteria

Selected gfx1100 model paths run with native rows plus explicit minimal deltas. No hidden stock/libref fallback and no wholesale generated rewrite.

## Notes

Execution detail and source matrix: docs/reviews/radiance-native-libr3d/ASSESSMENT.md. Do not infer support from patch applicability or an upstream row name.

At b8bca788 native h128 attention does not cover local Flash-Next h256/GQA6, and the actual Q27 INT8 fixture uses i8a8h rather than plain w8a8. Inspect selected rows, matrix operand packing, KV bytes and reachable reference gaps before designing each port. Preserve local higher-M and split-store behavior only where required.

## Change Log

- 2026-11-18T08:41:37.901602+00:00 (created-by): Created by agent
- 2026-11-18T08:49:08.981231+00:00 (updated-by): Updated: section:notes
