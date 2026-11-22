---
id: TRVP15
order: 0
plan: tuning-rocm-vulkan-provider
state: deferred
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

**Transferred TRVP14 acceptance (independent gate):** TRVP15's CM1 route qualification includes TRVP14 Gate 0 before any timing: use only pinned upstream native CM1 (do not transplant merged #27952/obsolete 1246) and define complete recipe identity for Q8_1 prep, CM1 main, optional split-K/reduce, op/quant/accumulator/shape/tile/subgroup/device/driver/ICD/shader digest/build/dispatch. Reuse TRBC03's Vulkan force interface and native can_execute/auto_eligible, never a parallel HIP ABI or shader pointer selector; strict supported force must prove actual pipeline bound with truthful post-bind activation, unsupported force must fail explicitly, auto/replay miss and device-lost/timestamp errors fall back visibly to stock. Global GGML_VK_DISABLE_COOPMAT is NOT an isolated CM1-off comparator. Test full-recipe parity before timing on S/M/L tails, quant, broadcast, grouped MUL_MAT_ID, Q8_1 prep and split/reduction. Preserve PRVP02's separate native CM1 A-prefetch end_k safety prerequisite; only with source/driver/physical-device/shader keyed correctness and ABBA+CI evidence may selected RDNA3/RDNA3.5/RDNA4 RADV/proprietary variants be promoted. Carry RDNA4 IQ4_XS and q4_1/q5_1/q4_k/q5_k/nvfp4 negative controls; Q2_K has no CM1 shader. Vulkan implementation remains PAUSED per RRVP02 until owner lifts the hold. No 1247 code/patch is authorised merely by merging plans.

## Notes

2026-10-08T22:03Z BCOP77: native CM1 is the pinned baseline, not an opt-in BigCherry backport. Preserve RDNA4 negative quant controls and include IQ4_XS; Q2_K has no CM1 shader. PRVP02 owns shader A-prefetch safety; TRVP14 owns any future route override. No Vulkan hardware qualification during the RRVP02 pause. Upstream and external benchmarks are not BigCherry promotion evidence.


Supersedes: RO22
Migration: capability-rebaseline-v3-2026-09
Successor key: tuning-rocm-vulkan-provider-ro22

Supersedes RO22. Consumes TRVP13 and TRVP14; preserve RD08 dependency/reference and do not claim a global architecture winner.

CONSOLIDATION 2026-11-22 PHASE-5: receive TRVP14. TRVP15's CM1 qualification cannot be meaningful without TRVP14's forceable, actual-bound CM1 recipe identity and telemetry; both refer to the same existing native Vulkan selector and one optional candidate family. Original acceptance still applies; the transferred gate is unfinished. Source is superseded for duplicate planning ownership, not completed on implementation, benchmark or live-hardware criteria. Where Vulkan is involved, implementation remains paused by owner directive.

DEFERRED VULKAN PORTFOLIO 2026-11-22: Explicit user-directed Vulkan implementation pause from 2026-09-10 remains in force. CM1 RDNA promotion depends on a native forced-route and safety proof that is not currently authorised under the Vulkan pause. This item is parked as deferred, NOT completed, superseded, deprecated, implemented or promoted; its original source/negative/hardware acceptance remains mandatory. Resume only via Planning API state->pending after a new owner review confirming: Resume after RRVP02 unpause, PRVP02 native CM1 end_k safety and a strict observed post-bind route, and TRVP12 runtime identity. Require same-device/driver/shader/shape correctness, complete prep cost and balanced ABBA CI evidence; keep unqualified candidates default-off. Shared HIP/ROCm identity, SDK, provider capabilities and benchmark controls remain active under their separate owners; this deferral does not block them.

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

## Reviews

- RV4531
- 2026-11-22T11:09:17.455813+00:00 (updated-by): Updated: section:acceptance_criteria, section:notes
- RV4569
- 2026-11-22T13:06:08.667821+00:00 (updated-by): Updated: section:notes
- 2026-11-22T13:11:14.275799+00:00 (state-transition): State: pending → deferred
