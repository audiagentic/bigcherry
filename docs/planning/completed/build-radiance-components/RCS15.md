---
id: RCS15
order: 0
plan: build-radiance-components
state: completed
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

Record exact candidate and source matched arms, CMake target and actual provider/row, model encodings including local u6/u7, L15/16, chained/bias, group128 vs 256, shape/scale/relayout/unrelayout, canonical/byte and measured speed. Mark unsupported separately; native scalar IQR not assumed optimized. RCS25 discrete first, existing mixed model gates later. Decision per covered path, no forced all-native switch.

## Notes

Execution detail and source matrix: docs/reviews/radiance-native-libr3d/ASSESSMENT.md. Do not infer support from patch applicability or an upstream row name.

Use tracked kbench-libr3d evidence as fixture history. The archived reports inspected use only libr3 and libref. They do not establish a native A/B baseline. Codeberg DNS/browsing did not provide a fresh HEAD during this assessment. Refresh and re-diff libr3d before taking implementation results.

Execution brief with the stage breakdown for RCS15 to RCS18: the libr3d native migration brief (briefs folder, libr3d-native-migration). Sequence agreed 2026-11-25: finish the gfx1201 fused-tile integration first (RCS19 first customer), then this baseline, GPU-free parts first (refresh the upstream pin, inspect the native provider's rows, inventory local optimisations and classify each), then the isolated arms A (generated gfx1100 library), B (stock native), C (reference oracle) on gfx1100 with the IQR rows first. Decision at the end of this item: does a native RDNA3 provider match the generated tree. Until then no new large generated-library projects except production fixes. The kernel-composition design (docs/design/radiance-kernel-composition.md, RCS19 to RCS22) stands either way for its first two layers. GPT is used for review only on this work; code it supplies comes back as text.

2026-11-25 part one done (GPU-free): docs/evidence/2026-11-25-rcs15-libr3d-inventory. Finding: at the pinned release (2.2.0, newest tag) upstream's native RDNA3 library has 51 units of its own, dot2 and wave-matrix arms per GEMM row, and NO row for the IQR (trellis 6/7-bit) form or the local int8 form with rotated activations (IQ8R); libr4d has IQR in 24 files. So a native-provider cutover cannot replace our gfx11 IQR and IQ8R path: those rows would have to be added, which is what the owned overlay plugins (RCS23) are, under either outcome of this item. The decision this item still owns is about the remaining rows (MXFP4, E4M3, GDN, MoE, DFlash, attention, helpers): matched arms A (our libr4d-units build), B (stock native library) and C (reference oracle) on one XTX, with IQR and IQ8R marked no native row.

2026-11-25 second integration review: docs/design/radiance-integration-review-20261125.md is the exact-source review and controls conflicting earlier wording. Shipping pin remains b6b8751 (2.2.0), upstream observed main 1c104de4 is a development candidate, 119 commits and 327 changed files; no automatic repin or backport. No new compatibility layer, launch manifest, bump driver or identity machinery. The pinned inventory no-IQR/no-native-int8 finding is valid for 2.2.0 only. Candidate aac92d7a/88b5a04a implements scalar packed IQR U3/U4/U5 L16 run128/256, group128, BF16 group or factored F32 scales, actual CMake only exact R3D_GPU_TARGETS=gfx1100, optional operands absent. Native int8 475e2fb3/5da0007e group128, not blanket IQ8R group256. u6/u7 and L15 not supported by that row. Native h128/KV work not proof h256/GQA6. Read exact model encoding and winning rows. Compare on one frozen source A generated/BC, B upstream libr3d, C reference; current shipped baseline separate. Candidate source test not yet built. Do not backport or select production based on row name.

Concurrent #1563 updated the existing RCS15 inventory with the same 119-commit main and advisory pin triage (no drift/coupling failures, patches affected, 48 target files and six rewrite-anchor files changed). Reuse that evidence, not a second inventory task. Our exact layout/source restrictions in the second integration review narrow the comparable subset. Main remains evidence only under tagged-release policy unless lead explicitly authorizes an isolated comparison build; no production repin. Actual composed patch check remains open.

2026-11-25 size of a bump to upstream main (tip 1c104de4, 119 commits past the 2.2.0 pin; no newer tag yet). The composition check applied the 35 production patches patch by patch on that tree: 24 apply unchanged, 11 do not (some are cascades of an earlier failure). Failing, in four clusters: (1) the tensor-parallel ownership family, rad0018, rad0020, rad0029, rad0048 (core, abi and arch/common: the declaration spans in the gated attention, delta-net and mlp blocks changed upstream; the others cascade from rad0018); (2) the b16 KV family, rad0051 to rad0054 (rad0051's anchor now matches twice; rad0052 fails on the qwen35 and qwen4exp architecture files, the gqa8 and h256 gqa6 kernels and the plugin header, which stops rad0053's new files and the two libr3 b16 rewrite files from existing): the heaviest cluster; (3) the prefill governor, rad0036 (scheduler); (4) the int8 pipelined prefill, rad0030 (the q8_fed branch). Also rad0065 (the gfx1201 fused tile) needs its template anchors re-pointed because upstream changed the prefill header. Gate 1 against the tree with the 24 applied patches: the only libr3 rewrite drift is the two b16 rewrite files, and only because rad0052 and rad0053 did not apply; every other rewrite block (8a8 prefill header, iqr, fp8a8, rows, selftest, all-reduce entry, fused quant) still matches. Gate 2a on gfx1201 for the units the 24 touch: 244 syntax passes clean. So a bump means re-porting 11 patches in four clusters, the b16 KV family first. Upstream has not tagged a release since 2.2.0, so nothing is started.

2026-11-25 part two (IQR row): upstream libr3d at 1c104de4 implements the IQR GEMM for gfx1100 as a scalar kernel (one thread per output column and row, trellis state rebuilt per element, no tiling or matrix instructions) on its own weight header format, not our IQR container, so byte comparison does not apply. Timed on the XTX cold (256 MiB flush, 4 bit, bf16 group scales): N=6144 K=5120 M=129 270 ms against 317 us ours (850 times slower), M=2048 2447 ms against 1.48 ms (1650 times); N=13312 M=129 678 ms against 625 us (1090 times). Decision: no reason to adopt upstream's gfx1100 IQR path; it stays a correctness reference. The int8 trunk GEMM and the 128x128 tile were not timed (no harness row call for them); RCS16 to RCS18 stay conditional on a measured native win and none is expected for IQR.

## Change Log

- 2026-11-18T08:41:29.761201+00:00 (created-by): Created by agent
- 2026-11-18T08:49:00.279834+00:00 (updated-by): Updated: section:notes
- 2026-11-25T00:41:57.599407+00:00 (updated-by): Updated: section:notes
- 2026-11-25T01:28:14.342119+00:00 (updated-by): Updated: section:notes

## Reviews

- RV4627
- 2026-11-25T02:37:49.407812+00:00 (updated-by): Updated: section:notes, section:acceptance_criteria
- 2026-11-25T02:54:55.206737+00:00 (updated-by): Updated: section:notes
- 2026-11-25T05:15:32.123253+00:00 (updated-by): Updated: section:notes
- 2026-11-25T08:45:39.280586+00:00 (updated-by): Updated: section:notes
- 2026-11-25T09:43:20.916233+00:00 (state-transition): State: pending → completed
