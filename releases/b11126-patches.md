# Release patch set

Selection: --source bigcherry

- **bigcherry revision:** 47b4795ca6086c9a5885ea55fa2f5d3f9598e13b
- **llama.cpp revision:** b1ff4ca23630ac7c5a275405353ddb5c486332c9
- **source:** bigcherry
- **target:** b11126

8 patch(es) included.

---

# 0100_cmake_options: CMake options for HIP replay/serving build (HI02, PA27)

**Status:** validated
**Plan item:** none

## What it does

Adds GGML_HIP_DISPATCH_REPLAY and related build options to ggml/CMakeLists.txt
(with configure-time validation) and turns them into HIP-backend compile
definitions plus the production dispatch/replay source list in
ggml/src/ggml-hip/CMakeLists.txt. Sets the `_BC_HIP_SERVING_BUILD_PLUMBING`
non-cache marker that `0110_campaign_tune_record_build` reads to fail closed
if its own options are activated without this package applied.

PA27 narrowed this package to only what a replay/serving build needs; the
tune/record-only options, tuner source, and campaign-only diagnostics moved
to `0110_campaign_tune_record_build`. There is no SQLite link edit anywhere
in the split (the declaration-only, no-consumer GGML_HIP_AUTOTUNE_SQLITE
option this package used to carry was dead code and was removed).

## Why

Measured dispatch needs its own build switches, and an illegal build
combination must fail at configure time rather than produce a silently
incomplete or inert build: GGML_HIP_DISPATCH_REPLAY=ON with GGML_HIP=OFF, and
dispatch combined with GGML_CUDA_FORCE_MMQ/GGML_CUDA_FORCE_CUBLAS (which
would hide candidate families from measurement).

## Upstream / provenance

Local design, part of this project's own HIP measured-dispatch framework (HI02).

---

# 0200_dispatch_hook: Route the dense matmul selector through measured dispatch (HI04)

**Status:** validated
**Plan item:** none

## What it does

Inserts a single guarded hook (ggml_hip_dispatch_mul_mat) at the top of upstream's ggml_cuda_mul_mat entry points; the hook returns false whenever it declines, so upstream's own ladder runs untouched. Also exposes the previously-static cuBLAS entry point so the BLAS candidate can reach it.

## Why

Upstream's selector decides and launches in one motion, so there is nothing to measure, store, or replay. A minimal, appended hook keeps the diff tiny and durable across releases while guaranteeing the native fallback is upstream's real code, not a reimplementation.

## Upstream / provenance

Local design, part of this project's own HIP measured-dispatch framework (HI04).

---

# 0300_mmq_forced_j: MMQ forced-J variant dispatch (HI06)

**Status:** validated
**Plan item:** none

## What it does

Splits upstream's mul_mat_q_switch_J into a scan (mul_mat_q_compute_J_best, lifted unchanged) and a launcher (mul_mat_q_launch_forced_J) that takes J as an explicit parameter, so a forced value can override the scan's answer while the native path stays identical.

## Why

The tuner needs to select and measure a specific MMQ tile width J instead of only ever seeing upstream's own scanned choice; separating the scan from the switch is the least invasive way to do that since launch_mul_mat_q already templates on J.

## Upstream / provenance

Local design, part of this project's own HIP measured-dispatch framework (HI06).

---

# 0400_mmvf_forced_block: MMVF forced block-size and accumulator-mode dispatch (HI07)

**Status:** validated
**Plan item:** none

## What it does

Threads a forced block-size/accumulator-mode value down through appended, defaulted parameters from ggml_cuda_mul_mat_vec_f to its launcher, touching only the call chain a forced value actually travels (replace_all edits with asserted match counts).

## Why

An earlier thread-local-override design was rejected because production replay builds would pay a per-launch read on the hottest path for a value that's always zero in production; an explicit parameter keeps the native path byte-identical to upstream.

## Upstream / provenance

Local design, part of this project's own HIP measured-dispatch framework (HI07).

---

# 0500_mmf_forced_nwarps: MMF forced-nwarps dispatch (HI08)

**Status:** validated
**Plan item:** none

## What it does

Same explicit-appended-defaulted-parameter shape as HI07, applied to MMF's three dispatchers (which share an identical signature/call tail); shared-memory sizes are recomputed from the forced nwarps immediately after the scan so allocation stays correct.

## Why

Needed so the tuner can force and measure a specific MMF nwarps value while leaving the native path byte-identical to upstream, without under-allocating shared memory for a forced value larger than native's choice.

## Upstream / provenance

Local design, part of this project's own HIP measured-dispatch framework (HI08).

---

# 0600_mmvq_geometry: Explicit MMVQ geometry variants (HI09 part 1)

**Status:** validated
**Plan item:** none

## What it does

Adds two defaulted template parameters (nwarps_explicit, rows_per_block_explicit) to the MMVQ kernel template; zero means derive geometry as upstream does (native instantiations unchanged), non-zero compiles a new geometry instance. Bounds are static_assert-checked in-kernel as a backstop.

## Why

MMVQ derives its geometry from calc_nwarps/calc_rows_per_block at compile time, so an alternative geometry needs genuinely new compiled code rather than a runtime switch, unlike MMQ/MMVF/MMF.

## Upstream / provenance

Local design, part of this project's own HIP measured-dispatch framework (HI09).

---

# 0650_mmvq_native_variant: Route a forced MMVQ geometry to its compiled instance (HI09 part 2)

**Status:** validated
**Plan item:** none

## What it does

Threads a forced-geometry struct down the existing native chain (ggml_cuda_mul_mat_vec_q -> mul_mat_vec_q_switch_ncols_dst) to the point where quantization/strides are already computed, diverging only at the launch call via ggml_hip_mmvq_find_instance.

## Why

Makes the geometry variants compiled by patch 0600 actually reachable, without duplicating upstream's quantization/stride logic (which would drift silently on every release). Refuses an unmatched geometry, MUL_MAT_ID width>1, and leaves fusion to the resolved instance rather than the forced path.

## Upstream / provenance

Local design, part of this project's own HIP measured-dispatch framework (HI09).

---

# 0700_coverage_counters: Family-entry instrumentation and coverage counters (HI13)

**Status:** validated
**Plan item:** none

## What it does

Adds counters at every real family entry point (not just the dense selector) to measure what fraction of matmul launches actually reach measured dispatch, since the graph optimizer calls MMVQ/MMVF directly for fused patterns, bypassing the dense selector.

HI168: counter calls and their reentrancy probes are compiled only with
`GGML_HIP_DISPATCH_DIAGNOSTICS`. Production retains family dispatch collection
without diagnostic counting. The upgrade edit also guards previously applied
hooks. This removes known instrumentation work; throughput parity still requires
a controlled hardware comparison.

## Why

Without this number, a tuning run's coverage of real model work is unknown, and 'we tuned the model' is an unverified assumption; test-backend-ops cannot produce this figure since it bypasses the graph optimizer entirely.

## Upstream / provenance

Local design, not in the original plan; added to answer a coverage question no other patch answers (HI13).

---
