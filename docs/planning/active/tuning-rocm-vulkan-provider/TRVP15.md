---
id: TRVP15
order: 0
plan: tuning-rocm-vulkan-provider
state: pending
created-at: '2026-09-09T11:01:20.673012+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: P1
---

# Qualify and selectively promote RDNA CM1 winners

## Description

Qualify and selectively promote any *additional* RDNA CM1 route/recipe winner on exact device/driver/signature identities. Upstream CM1 is already native and auto-eligible at b11474; only new BigCherry overrides remain disabled absent evidence.

## Steps

1. Run native and forced CM1 signatures on gfx1100, gfx115x, and gfx120x, with RADV and proprietary AMD stacks separately. 2. Require correctness for all supported quant types in MUL_MAT and MUL_MAT_ID, including tails, split/non-split K, batched shapes, expert routing, and preparation/reduction. 3. Measure forced candidates under statistical/effect-evidence policy with warm-up excluded and full recipe timing. 4. Tune per exact device/driver/shader/build/signature key; never promote architecture-wide aggregates. 5. Seed RDNA4 negative controls q4_1, q5_1, q4_k, q5_k, nvfp4 MUL_MAT and nvfp4 MUL_MAT_ID. 6. Promote only replay-safe winners with complete identity; stale device/driver/shader/vendor-pin/candidate/signature misses fall back to native. 7. #27952 is already merged into the b11474 pin: never port 1246. First close PRVP02's A-prefetch end_k correctness gate; retain native shader as control and only qualify separately identifiable new recipes.

## Detailed Solution & Technical Design

Treat CM1 gains and mixed RDNA4 results as hypotheses. Qualification order is gfx1100, gfx115x, gfx120x, each stack separately. Promotion key is exact physical device, Vulkan driver/API/ICD, shader/SPIR-V digest, BigCherry/vendor build, dispatch signature, and complete recipe. Operator timing and end-to-end MoE results can diverge, so per-signature evidence precedes PP/TG aggregates. Use TRVP13 inspection/replay gates and existing workflow evidence; this is a qualification campaign, not a fourth source patch.

## Code Samples & Guidance



## Files

Vulkan campaign/tune/replay evidence; candidate promotion/replay artifacts; per-device/driver benchmark logs; correctness reports; PRVP01-TRVP14 outputs; existing experiment contract/evidence and promotion modules.

## Validation

Require exact output parity before timing. Cover S/M/L, aligned/tail M/N, multiple K, batched/broadcast, split/non-split, contiguous F32-to-Q8_1 preparation, ineligible activation, and MUL_MAT_ID expert routing. Verify full recipe timing excludes shader compilation/pipeline creation/first-run warm-up unless contract says otherwise. Replay exact identity hits CM1; any changed driver/device/shader/vendor pin/signature misses to native. Record source/build manifests, telemetry, shader audit, correctness, measurements, promotion, replay diagnostics, and negative-control results.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Only exact-identity, correctness-passing, statistically supported winners are promoted; default enablement stays off absent evidence. Native fallback is visible and reliable for all misses/failures. RDNA3, RDNA3.5, and RDNA4 plus RADV/proprietary stacks are not conflated, negative controls are retained, and complete campaign lineage is reproducible.

## Notes

2026-10-08T22:03Z BCOP77: native CM1 is the pinned baseline, not an opt-in BigCherry backport. Preserve RDNA4 negative quant controls and include IQ4_XS; Q2_K has no CM1 shader. PRVP02 owns shader A-prefetch safety; TRVP14 owns any future route override. No Vulkan hardware qualification during the RRVP02 pause. Upstream and external benchmarks are not BigCherry promotion evidence.


Supersedes: RO22
Migration: capability-rebaseline-v3-2026-09
Successor key: tuning-rocm-vulkan-provider-ro22

Supersedes RO22. Consumes TRVP13 and TRVP14; preserve RD08 dependency/reference and do not claim a global architecture winner.

## Change Log

- 2026-09-09T11:01:20.673012+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:18:18.184715+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.605439+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:47.476418+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:47:03.253667+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria, section:notes
- chg_20260910_034835_repaired-the-final-six-live-su_3009
- 2026-09-10T03:48:35.262866+00:00 (updated-by): Updated: section:ledger-events
