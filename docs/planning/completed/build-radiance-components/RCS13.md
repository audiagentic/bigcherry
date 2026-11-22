---
id: RCS13
order: 0
plan: build-radiance-components
state: superseded
created-at: '2026-11-13T03:53:14.434848+00:00'
breadth: ''
skill: advanced
created-by: bigcherry-1
priority: P0
work: M
---

# C02: GEMV K bound removed - launch-sized LDS and pre-launch dispatch to grouped GEMM

## Description

Umbrella: RCS12 (MC-004/005; audit C02). The remote-MoE GEMV kernels (`moe-remote/moe_lane_gemm.h`, `bc-gfx12/bc_moe_gemv.hip`) hold activations in fixed LDS arrays sized for K <= 2560 and return early above it; the dispatcher picks them by row count only, and `Runner::set_shape` validates hidden but not intermediate, so a model with intermediate > 2560 can report success with no fresh output. Replace the bound with launch-sized dynamic LDS plus a checked, pure per-variant LDS calculator, and choose GEMV or the existing grouped GEMM before launch. Analysis: GPT 2026-11-13 (kept in this item; source brief in the work root).

## Steps

1. Card-free first: `moe-remote/gemv_geometry.h` (pure host-readable): `Variant` enum for the compiled variants (generic fp8/w4, wide, H2 R1/R2/R4, W4 H2, multi-row R4, gfx12 R2), `required_lds_bytes(variant, K, group)` with checked arithmetic, alignment and actual group count, and `choose_gemv(variant, K0, K1, capabilities)` returning Gemv{variant,dynamic_lds} | GenericGemm | Refuse{reason}. Register `gemv_geometry_test` in CMake and `component.toml`.
2. Kernel bodies: replace `kGemvMaxK` static arrays at `moe_lane_gemm.h:36,47-48,144-145,253-255,376-378,469-474` with one aligned `extern __shared__` region and typed pointers from the host descriptor; keep weight indexing, conversion, dot instruction, per-wave summation order, BF16 rounding, `__launch_bounds__(256)` and the bank layout unchanged. Remove the `K > 2560` early returns (`:58,155,268,388,469`) only once host eligibility is authoritative.
3. gfx12: same change in `bc-gfx12/bc_moe_gemv.hip:23,57-59,71-86,189-210,229-240`; the existing host `RAD_E_UNSUPPORTED` gate uses the computed fit; `bc_gfx12_plugin.cpp:120-126` still delegates to libr4d and fails when neither can run.
4. Dispatch: `moe_remote_gpu.h:27-57` `launch_gemv` returns a checked selection; decide per stage (gate/up K=hidden, down K=intermediate) before launch; the generic grouped GEMM is the fallback; a genuine launch failure fails closed. `set_shape/open` (`:101-109,149-179`) validates both K values against the device (`hipGetDeviceProperties`); refuse at startup only when a required stage has no runnable implementation. `mid_w()` stays `shape_.intermediate`.
5. Delete both category-C allowances in `engines/radiance/model_constant_guard.toml:17-22,80-85` in the same PR and fix the stale row descriptions at `:11-16`.
6. No K tiling in this change (it changes accumulation order and costs barriers); a tiled GEMV is a separate evidence-driven item only if large-K decode regresses.

## Detailed Solution & Technical Design

Fixed values that stay: FP8 scale group 128, vector width 16, W4 group 64, 256-thread launch, row batch R=1/2/4. Other group sizes or formats refuse this GEMV and use a verified kernel. Parity-specific K: allocate for max(K0,K1), index each class by its own K. Fallback is decided before any GEMV launch; an error after launch is never converted into an alternate kernel, so a partial writer cannot corrupt the fallback. Same math in source does not guarantee the same ISA or bytes: identity is a test gate, not an assumption; if the compiler reorders, block until the hardware test shows it.

## Code Samples & Guidance



## Files

engines/radiance/components/moe-remote/{moe_lane_gemm.h,moe_remote_gpu.h,gemv_geometry.h (new),gemv_geometry_test.cpp (new),CMakeLists.txt,component.toml}, engines/radiance/components/bc-gfx12/{bc_moe_gemv.hip,bc_gfx12_plugin.cpp}, engines/radiance/model_constant_guard.toml.

## Validation

Card-free (no lease): gemv_geometry_test over every variant and both projection Ks with K below/at/above 2560 (a regression fixture only), K=0, misaligned, overflow, capabilities below/equal/above; fake-capability dispatch tests (hidden=2560 intermediate=3072 never selects the small-lane down GEMV; hidden>2560 with a generic GEMM is not refused for width; both kernels rejecting returns a typed error before publish); CPU exact-order oracle for index math (it cannot prove GPU bytes; existing `moe_fp8_ref_test` does not exercise these kernels); `tools/tests/core/test_model_constants.py`; mutation checks (restore a K>2560 return, drop the down-K check, wrong LDS formula) must fail; existing r3_moe_fp8_ref_test, r3_moe_backend_test, r3_worker_footprint_test; HIP compile-only for gfx1201, gfx1100, gfx1030 (spills, dynamic LDS attributes).
Hardware (short gpu leases, per arm): A) one captured Flash-Next helper request via `helper-capture.sh` + `r3_moe_worker --replay/--compare` and `w4-replay-hash.sh`: K=2560 result buffers byte-identical to the same variant before the change; B) `kbench_ab.py` `moe_gemm_q` gfx1201 A/B/A with checked `rad-kbench --report`; C) production parity with `helper-ab.json` and `helper-kld.json`, decode ms/step at 1 and 6 streams and prefill TTFT reported separately. STOP on any byte mismatch, sentinel/unwritten result, silent success on ineligible K, or decode slowdown beyond the A/B/A spread.

## Effort & Risk

M. Risks: dynamic LDS changing codegen or occupancy (MR=4 already uses 40 KiB at K=2560); LDS stride/padding must match device pointer arithmetic exactly (H2 two planes, wide four planes, 16-byte loads); not measured: sharedMemPerBlock, per-kernel LDS, VGPR/SGPR and occupancy on the installed driver for the three ISAs; a generic grouped GEMM row must exist for every format and ISA before a manifest may start.

## Standards

No hand-picked constants; no compatibility shim; kernel changes only as bigcherry components or managed patches via slice/PR; fail closed; migrate every caller in the same change.

## Acceptance Criteria

1. No K-bound literal in GEMV kernels or dispatch; both guard allowances removed. 2. A model with intermediate > 2560 either runs through the grouped GEMM or is refused at startup with stage/K/reason, never a no-op success. 3. K=2560 outputs byte-identical per variant and no attributable decode regression on gfx1201, gfx1100, gfx1030 (each qualified independently). 4. Card-free tests and mutation checks in CI.

## Notes

Stage 1 of the radiance protocol programme (see RCS12 staging): independent of the shared-memory ABI, so it can land first and in parallel with RHP01 evidence work.

Stage 1 of the radiance protocol programme (see RCS12 staging): independent of the shared-memory ABI, so it can land first and in parallel with RHP01 evidence work.

Status 2026-11-18: implementation merged to main with PR 946; the item stays in progress until criterion 2 (a model with intermediate > 2560) and criterion 3 on gfx1100 and gfx1030 are qualified. This is a correctness change, not a speed change: neutral performance is the expected and acceptable result.

Evidence (docs/evidence/2026-11-17-rcs13-pr946-run6 .. 2026-11-18-rcs13-pr946-run12; runs 6-11 are gate-stop records):
- Offline: K=2560 replay result buffers byte-identical (8/8 captured requests); theoretical W4H2 occupancy 8 -> 7 blocks per CU on gfx1201 (not a runtime result).
- Run 12 (production launcher, TP2 gfx1100 engines + gfx1201 helper, A = main 7c0273976, B = PR 946 head, A/B/B/A, three seeds, 1 and 2 decode streams, separate prefill cells): decode engine ms/step A 37.85 and 37.70, B 37.53 and 36.96 (pooled B -1.4%, inside the spread: A repeats differ by 0.4%, B repeats by 1.7%); prefill TTFT A 1410 and 1415 ms, B 1402 and 1441 ms (+0.6%, inside B's own 2.7% spread); MTP draft/accept equal within noise (A 479/702, B 480/696); single-stream output hashes identical in all four cells; two-stream hashes differ across cells and also between two runs of the same arm A (concurrency nondeterminism present without the patch); 6488 executions of the dynamic-LDS W4H2 kernel recorded in B. Verdict recorded as: no attributable regression, no improvement claim; the strict multi-stream identity gate cannot be met by any arm and needs a deterministic concurrency oracle.
- Process findings of runs 5-11 (all on main, fixed): TP2 helper footprint drift (60 B per lane, PR 1056); Python planner exit-11 remeasure and the S10 resolver rewriting the helper budget after measurement (PR 1058); helper VRAM release barrier before the final manifest (PR 1061).

Card-free large-K model admission (2026-11-18): the existing declared `r3_gemv_geometry_test` already exercised the individual stage selections for 2560/3072, hidden=4096 with grouped-GEMM fallback, and typed `Refuse` without a fallback. A model-pair regression now explicitly checks hidden=2560/intermediate=3072 together: gate/up GEMV versus down grouped GEMM under a 2560-sized LDS limit; hidden=4096 with grouped GEMM; typed down refusal when neither implementation can run; and a launch-sized down GEMV when 3072 actually fits. CTest gives that target a 60-second timeout. Source audit: `Runner::open` calls `set_shape` before lane preparation; `enqueue_lane` selects both stages and checks `Refuse` before launching stage outputs or `lane_publish_done`. This is selector-level card-free coverage, **not** a test of a live startup refusal or GPU output identity.

Open: (1) **hardware** acceptance of an actual model with intermediate > 2560, including grouped-GEMM fallback or typed startup refusal and non-stale outputs; (2) gfx1100 and gfx1030 byte identity and decode parity, each qualified independently; (3) a deterministic multi-stream oracle if a concurrency identity gate is wanted.

CONSOLIDATION 2026-11-22 NATIVE-OWNERS: C02 GEMV K-bound implementation and guard expiry have one canonical owner under RCS12 instead of an audit ticket and a second kernel implementation ticket. Canonical remaining owner: RCS12. RCS13 identifies itself as the RCS12 MC-004/005 (C02) execution fix; RCS12 already names this exact guarded implementation requirement, so keeping two plan owners duplicates the same kernel admission task. Exact remaining acceptance and stop/fallback requirements are in the active receiver. This lifecycle state retires duplicate planning ownership only; original source/evidence/negative results are retained and not declared complete.

## Change Log

- 2026-11-13T03:53:14.434848+00:00 (created-by): Created by bigcherry-1
- 2026-11-13T03:53:59.413255+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:effort_risk, section:standards, section:acceptance_criteria, section:notes
- 2026-11-17T22:25:51.124293+00:00 (updated-by): Updated: section:notes
- 2026-11-17T22:26:00.089362+00:00 (state-transition): State: pending → in_progress

## Reviews

- RV4508
- 2026-11-22T10:36:51.300961+00:00 (updated-by): Updated: section:notes
- 2026-11-22T10:43:57.520227+00:00 (state-transition): State: in_progress → superseded
