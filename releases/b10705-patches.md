# Release patch set

Selection: --recipe bigcherry

- **bigcherry revision:** c8a3a445d9f65886cf9f608afa0ef8da1ed71a31
- **llama.cpp revision:** 2578138397d7b422bb0e160efdd429976c55fb55
- **recipe:** bigcherry
- **target:** b10705

15 patch(es) included.

---

# 0100_cmake_options: CMake options for HIP measured dispatch (HI02)

**Status:** validated
**Group:** core
**Plan item:** none

## What it does

Adds GGML_HIP_AUTOTUNE and related build options to ggml/CMakeLists.txt (with configure-time validation) and turns them into HIP-backend compile definitions plus SQLite linkage in ggml/src/ggml-hip/CMakeLists.txt.

## Why

Measured dispatch needs its own build switches, and two illegal build combinations must fail at configure time rather than produce a silently incomplete or inert build: GGML_HIP_AUTOTUNE=ON with GGML_HIP=OFF, and dispatch combined with GGML_CUDA_FORCE_MMQ/GGML_CUDA_FORCE_CUBLAS (which would hide candidate families from measurement).

## Upstream / provenance

Local design, part of this project's own HIP measured-dispatch framework (HI02).

---

# 0200_dispatch_hook: Route the dense matmul selector through measured dispatch (HI04)

**Status:** validated
**Group:** core
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
**Group:** core
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
**Group:** core
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
**Group:** core
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
**Group:** core
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
**Group:** core
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
**Group:** core
**Plan item:** none

## What it does

Adds counters at every real family entry point (not just the dense selector) to measure what fraction of matmul launches actually reach measured dispatch, since the graph optimizer calls MMVQ/MMVF directly for fused patterns, bypassing the dense selector.

## Why

Without this number, a tuning run's coverage of real model work is unknown, and 'we tuned the model' is an unverified assumption; test-backend-ops cannot produce this figure since it bypasses the graph optimizer entirely.

## Upstream / provenance

Local design, not in the original plan; added to answer a coverage question no other patch answers (HI13).

---

# 0800_server_shutdown_endpoint: Opt-in HTTP shutdown endpoint for graceful automation cleanup

**Status:** validated
**Group:** core
**Plan item:** none

## What it does

Adds a POST /shutdown endpoint, disabled by default and enabled via LLAMA_SERVER_ENABLE_SHUTDOWN, that returns 202 and then enters the server's normal termination path.

## Why

Windows benchmark harnesses that terminate the server process directly skip backend destruction and lose buffered HIP autotune measurements; a graceful shutdown path avoids that loss.

## Upstream / provenance

Local design, part of this project's own tooling/automation support.

---

# 0810_replay_hit_diagnostics: Optional, compile-time replay hit diagnostics

**Status:** validated
**Group:** core
**Plan item:** none

## What it does

Gates a hit-recorder and aggregated JSONL hit log behind GGML_HIP_REPLAY_DIAGNOSTICS/GGML_HIP_DISPATCH_HIT_LOG so production replay builds compile out the diagnostics branch and its synchronization cost entirely.

## Why

Diagnosing replay dispatch-table hits needs visibility, but that visibility must not cost anything in production replay builds.

## Upstream / provenance

Local design, part of this project's own HIP measured-dispatch framework.

---

# 0820_measurement_signature_shapes: Persist canonical signature shapes in tuning measurements

**Status:** validated
**Group:** core
**Plan item:** none

## What it does

Stores the canonical signature shape alongside each tuning measurement record.

## Why

Downstream tooling needs the canonical shape associated with a measurement, not just its raw dimensions, to correctly group and replay candidates.

## Upstream / provenance

Local design, part of this project's own HIP measured-dispatch framework.

---

# 0830_split_reduce_telemetry: Observe actual SPLIT_REDUCE provider and meta handoff (HI58)

**Status:** validated
**Group:** core
**Plan item:** none

## What it does

Instruments the SPLIT_REDUCE path to record which reduction provider (RCCL/meta) actually handled a given reduction and the handoff between them.

## Why

Needed to verify, from real telemetry rather than assumption, which reduction path a given multi-GPU run actually used.

## Upstream / provenance

Local design, part of this project's own telemetry work (HI58).

---

# 0900_pool_workspace_metrics: Measured per-candidate workspace via the pool's own bookkeeping (HI52 part 1)

**Status:** validated
**Group:** core
**Plan item:** none

## What it does

Adds a bc_requested_size counter to ggml_cuda_pool's base struct, incremented/decremented at the single choke point (ggml_cuda_pool_alloc's two call sites) where every pool alloc/free in the CUDA/HIP backend passes through, tracking each candidate's requested size rather than the pool's actual-size bookkeeping.

## Why

measurement.workspace_bytes previously reported each candidate's declared upper bound, which is constant within a family and has never discriminated anything (HI45's low-memory Pareto profile always reports 0% savings). A device-global hipMemGetInfo delta was tried and failed structurally because the caching pool reuses a high-water-mark allocation. The pool's own bookkeeping is the only place that actually knows the answer.

## Upstream / provenance

Local design, part of this project's own HIP measured-dispatch framework (HI52).

---

# 1000_rdna4_mmq_q2k_q6k_fix: Upstream backport: RDNA4 MMQ codegen fixes for Q2_K and Q6_K

**Status:** validated
**Group:** upstream-fixes
**Plan item:** none

## What it does

Cherry-picks two narrowly-scoped fixes from unmerged upstream PR #25940 into the MFMA/WMMA MMQ vec_dot for Q2_K (forces a plain loop via #pragma unroll 1 to avoid a ROCm over-unroll/spill) and Q6_K (adds an explicit float cast before a scale multiply to change ROCm's codegen).

## Why

The PR's own numbers show large RDNA4 gains (Q6_K 1.90x, Q2_K 28.2x at n=512), and both quant types are in this project's own test corpus and hardware, so the fix is worth taking ahead of upstream merge rather than waiting.

## Upstream / provenance

Cherry-picked from open upstream PR https://github.com/ggml-org/llama.cpp/pull/25940. Deliberately excludes the PR's second change (a hand-written RDNA4 native-select heuristic), since this project's own tuner already measures candidates head-to-head per shape.

---

# 1100_hi70_direct_op_evidence: Deterministic direct-op correctness corpus for hard-to-reach candidates (HI70)

**Status:** validated
**Group:** core
**Plan item:** none

## What it does

Adds a small, deterministic, CPU-referenceable test_mul_mat corpus to test-backend-ops that constructs the exact tensor shapes (fallback MMQ M%128!=0, and MMF f16 batch widths 1-16) that well-formed production models structurally never produce.

## Why

24 of gfx1201's 100 candidates never got correctness evidence from any real-model workload because reaching them requires shapes production models don't naturally hit; hunting for a lucky real model is not a reliable evidence strategy and ran out of local models for q5_k entirely.

## Upstream / provenance

Local design, part of this project's own correctness-evidence work (HI70), per gpt-auto-agent's deep-dive recommendation.

---
