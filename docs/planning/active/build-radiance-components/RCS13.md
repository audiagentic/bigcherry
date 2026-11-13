---
id: RCS13
order: 0
plan: build-radiance-components
state: pending
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

## Change Log

- 2026-11-13T03:53:14.434848+00:00 (created-by): Created by bigcherry-1
- 2026-11-13T03:53:59.413255+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:effort_risk, section:standards, section:acceptance_criteria, section:notes
