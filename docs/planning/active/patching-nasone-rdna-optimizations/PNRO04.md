---
id: PNRO04
order: 0
plan: patching-nasone-rdna-optimizations
state: in_progress
created-at: '2026-09-09T10:52:17.766702+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: P0
---

# gfx1100 BF16/WMMA fused chunked GatedDeltaNet

## Description

TODO, NOT-READY (rescoped). Verified via patch.py: 1253 only defines the eligibility predicate `bigcherry_nro04_gfx1100_bf16_ready()`, which unconditionally `return false;` -- no BF16/WMMA kernel exists and nothing is dispatched. A separate, already-implemented gating predicate (line ~22) uses `GGML_CUDA_CC_IS_RDNA3(cc)`, which is broader than exact gfx1100 (RDNA3 covers more than just gfx1100) -- if exact-gfx1100-only scope is intended, that predicate needs narrowing too.

## Steps

1. Materialize the reviewed BF16/WMMA GatedDeltaNet kernel into ggml/src/ggml-cuda/gated_delta_net.cu (verify this is the correct existing/target file at implementation time), reusing RD50's existing chunked-recurrence structure.
2. Wire dispatch from ggml_cuda_op_gated_delta_net_impl() (verify exact function name) BEFORE the sequential fallback, gated on the corrected eligibility check.
3. Fix `bigcherry_nro04_gfx1100_bf16_ready()` to return a real eligibility result once the kernel exists (it currently unconditionally returns false, so nothing can ever activate); if the intended scope is exact gfx1100 only, also narrow the existing `GGML_CUDA_CC_IS_RDNA3(cc)` gate (line ~22) accordingly, since RDNA3 is broader than gfx1100 alone.
4. Add an actual-launch activation trace marker (BIGCHERRY_PATCH_TRACE-gated) -- none exists today since nothing is ever dispatched.
5. Compile WMMA only in the gfx11 device pass and never cross-select gfx12; add opt-out/force control and preserve sequential fallback.
6. Validate WMMA fragment layout with deterministic identity/structured/random/adversarial matrix probes.
7. Compare one/multi-chunk recurrence, recurrent state, CPU/reference outputs, long-sequence PPL/KL/greedy quality, graph/non-graph execution.
8. Profile KKT/scan separately and promote only gfx1100 exact shape after quality and E2E evidence.

## Detailed Solution & Technical Design

Capability owner: patching

Split assessment: One independent boundary; Build/Run support is a dependency.

Overlap assessment: No duplicate boundary found; related items are prerequisites or adjacent evidence.

## Code Samples & Guidance



## Files

patches/1253_nro04_gfx1100_bf16_chunked_gdn; GDN translation unit/helpers; shared package tests; WMMA/GDN fixtures

## Validation

Patch mechanics: `PYTHONPATH=tools python -m bigcherry patch-lint patches/1253_nro04_gfx1100_bf16_chunked_gdn`; `PYTHONPATH=tools python -m bigcherry patch-rebase-check --focal-overlay 1253_nro04_gfx1100_bf16_chunked_gdn --source bigcherry-tuning`; package pytest offline (static arch-guard/non-gfx11 compile-exclusion checks). Deterministic WMMA fragment matrix probe (identity/structured/random/adversarial). Hardware (Brutus, gfx1100 required): `python -m bigcherry.patch.validation_campaign --overlay 1253_nro04_gfx1100_bf16_chunked_gdn --arch gfx1100` comparing one/multi-chunk recurrence vs CPU/reference, long-sequence PPL/KL/greedy quality, graph/non-graph execution, KKT/scan profiling; gfx1201/gfx1030 must remain non-selecting controls.

## Effort & Risk



## Standards

Correctness before performance; architecture-specific promotion; source SHA fixed; no synthetic inheritance of RDNA4 evidence.

## Acceptance Criteria

gfx1100 WMMA primitive and GDN/state pass registered tolerances; non-gfx11 devices compile and cannot select; real gfx1100 effect is positive without quality regression; fallback remains verified.

## Notes

Supersedes: NRO04
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-nasone-rdna-optimizations-nro04

Supersedes: NRO04
Inherited semantic scope: preserve gfx1100 geometry, WMMA fragment probe, FP32 state, architecture separation, and fallback gates.
Migration: capability-rebaseline-v3-2026-09

2026-09-24 relevance at b11126: IMPLEMENTED-AS-PATCH. patches/1253_nro04_gfx1100_bf16_chunked_gdn exists, state=untested. Builds on existing RD50 chunked recurrence (already in bigcherry patch history). No upstream gfx1100 BF16/WMMA GDN equivalent found relevant to this project's patch history. Disposition: validate/qualify existing patch; no GPT design needed.

2026-09-24 GPT review req_215c89d0b13a4bb7 applied: verified bigcherry_nro04_gfx1100_bf16_ready() unconditionally returns false and no BF16/WMMA kernel or dispatch exists in 1253 -- added the required kernel-materialization and dispatch-wiring steps plus a required activation marker (none existed). Flagged the separate GGML_CUDA_CC_IS_RDNA3 gate as broader than exact gfx1100 if strict scope is intended.

2026-09-25 (d423644d, f4c75c02): 1253 now ports nasone 4169fbbf (block 02) BF16/WMMA chunked GDN: creates gated_delta_net_chunked.cuh, _bf16_gfx11.cu (RDNA3), _bf16.cu (RDNA4) via new FilePatch(create=True); K==1 S_v==128 prefill on RDNA3/RDNA4 routes to BF16 by default (opt-out GGML_CUDA_GDN_CHUNKED_BF16=0), sequential fallback on launch rejection; fork GDN test cases added. fp32 chunked kernel NOT ported (RD50/1221 scope; reciprocal conflict, no longer requires 1221). Contract NRO04-GDN-CHUNKED-BF16 (prefill positive, decode control, owner policy x4 sessions); producer = GATED_DELTA_NET test-backend-ops backend_reference + marker + paired lanes. Hardware sessions not yet queued.

## Change Log

- 2026-09-09T10:52:17.766702+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:08:46.573893+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.067141+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.708292+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:26:15.401708+00:00 (updated-by): Updated: section:description, section:steps, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes
- chg_20260910_022800_five-nasone-successor-plans-no_4030
- 2026-09-10T02:28:00.297184+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T02:26:11.680244+00:00 (updated-by): Updated: section:validation, section:notes
- 2026-09-24T04:48:19.490128+00:00 (updated-by): Updated: section:description, section:steps, section:notes
- 2026-09-25T04:23:36.714622+00:00 (updated-by): Updated: section:notes
- 2026-09-25T04:23:39.602014+00:00 (state-transition): State: pending → in_progress
