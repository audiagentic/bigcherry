---
id: RCS15
order: 0
plan: build-radiance-components
state: pending
created-at: '2026-11-18T08:41:29.761201+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# Baseline upstream libr3d against the BigCherry RDNA3 library

## Description



## Steps

Refresh upstream and freeze source/ABI/runtime identities with RUM03 and BRBC05. Build both providers as isolated comparison arms from the same target revision if possible. Use KAB01 for comparisons and explicit kernel orders. Begin with one 7900 XTX and one approved dense fixture. Record selected rows, refusals and no-device-row cases before timing.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Native selftest against matched libref, local rewrite check for the comparison arm, KAB01 correctness and A/A noise, then matching dense decode/prefill timing. Reject mismatched schemas, library fallback or unbound artifact hashes.

## Effort & Risk



## Standards



## Acceptance Criteria

A reproducible matched-baseline row matrix separates native coverage, correctness gaps, performance gaps and untested shapes.

## Notes

Execution detail and source matrix: docs/reviews/radiance-native-libr3d/ASSESSMENT.md. Do not infer support from patch applicability or an upstream row name.

Use tracked kbench-libr3d evidence as fixture history. The archived reports inspected use only libr3 and libref. They do not establish a native A/B baseline. Codeberg DNS/browsing did not provide a fresh HEAD during this assessment. Refresh and re-diff libr3d before taking implementation results.

Execution brief with the stage breakdown for RCS15 to RCS18: the libr3d native migration brief (briefs folder, libr3d-native-migration). Sequence agreed 2026-11-25: finish the gfx1201 fused-tile integration first (RCS19 first customer), then this baseline, GPU-free parts first (refresh the upstream pin, inspect the native provider's rows, inventory local optimisations and classify each), then the isolated arms A (generated gfx1100 library), B (stock native), C (reference oracle) on gfx1100 with the IQR rows first. Decision at the end of this item: does a native RDNA3 provider match the generated tree. Until then no new large generated-library projects except production fixes. The kernel-composition design (docs/design/radiance-kernel-composition.md, RCS19 to RCS22) stands either way for its first two layers. GPT is used for review only on this work; code it supplies comes back as text.

2026-11-25 part one done (GPU-free): docs/evidence/2026-11-25-rcs15-libr3d-inventory. Finding: at the pinned release (2.2.0, newest tag) upstream's native RDNA3 library has 51 units of its own, dot2 and wave-matrix arms per GEMM row, and NO row for the IQR (trellis 6/7-bit) form or the local int8 form with rotated activations (IQ8R); libr4d has IQR in 24 files. So a native-provider cutover cannot replace our gfx11 IQR and IQ8R path: those rows would have to be added, which is what the owned overlay plugins (RCS23) are, under either outcome of this item. The decision this item still owns is about the remaining rows (MXFP4, E4M3, GDN, MoE, DFlash, attention, helpers): matched arms A (our libr4d-units build), B (stock native library) and C (reference oracle) on one XTX, with IQR and IQ8R marked no native row.

## Change Log

- 2026-11-18T08:41:29.761201+00:00 (created-by): Created by agent
- 2026-11-18T08:49:00.279834+00:00 (updated-by): Updated: section:notes
- 2026-11-25T00:41:57.599407+00:00 (updated-by): Updated: section:notes
- 2026-11-25T01:28:14.342119+00:00 (updated-by): Updated: section:notes
