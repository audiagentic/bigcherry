---
id: BCOP227
order: 227
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-15T07:08:38Z'
created-by: agent
priority: P2
work: S
---

# PRBE70: CK GEMM oracle physical-layout and output-dtype fence

**Discovery:** Pinned ggml MUL_MAT stores F32 `dst[O,T]` with O contiguous. The existing CK profiler example used `A=weight[O,K], B=activation[T,K], C[O,T]` and would miscompare flattened output on nonsquare cases because CK C is row-major. Canonical zero-copy screening swaps A/B: `A=activation[T,K], B=weight[O,K], C[T,O]`. CK F16/BF16 profiler GEMM outputs narrow F16/BF16, not ggml's F32: do not claim native output parity or zero-cost substitution.

**Owner/consolidation:** PRBE70 alone owns offline CK oracle admission; PRBE28 float source weights, PRBE29/30 F16 shadows/padding and PRBE31 BLAS crossover own any downstream implementation. No new kernel, runtime dispatcher, allocator, scheduler, experiment queue or telemetry. Active MEN10/Radiance/RHP/RWQ/PGC15 remain untouched.

**Eligibility/novelty:** `main` selected at independent `08441aa4a69e34d6905e6449e19467900ae19c95` (2026-10-15 05:21 UTC, MEN10/RAD09). PRBE70's latest relevant independent work predates the 12-hour window; no matching recent PR, patch, benchmark, queue or hardware receipt found. Recent BCOP219-226 concern other capabilities. This audit adds a new physical-layout/dtype discrepancy not covered by PRBE70's 2026-10-09 typed-manifest review.

**Evidence:** b11552 `ggml.c::ggml_mul_mat`, `ggml-cuda.cu::ggml_cuda_mul_mat_cublas_impl`; ROCm CK `profiler/src/profile_gemm.cpp`. Five host-only Python tests passed (non-square flat-order counterexample, corrected mapping, shape, F16 rounding and shape sweep). No repository suite, CK binary, HIP build, GPU test or performance benchmark run; BigCherry CK gain unmeasured.

**Unresolved terminal:** one source-typed existing BLAS signature and CK instance preflight; reject unknown/batched/padded shapes. Verify A/B swap and output dtype; F16/BF16 `OUTPUT_TYPE_MISMATCH` stays screening-only with F32 cast cost. Only a physically measured >=3% exposed E2E ceiling and compatible typed output can reach existing owner. Otherwise close `CK_UNAVAILABLE`, `UNKNOWN_SIGNATURE`, `OUTPUT_TYPE_MISMATCH`, `NO_EXPOSED_HEADROOM` or `NO_PORT`.

**Sources:** https://github.com/ROCm/rocm-libraries/blob/develop/projects/composablekernel/profiler/src/profile_gemm.cpp ; https://github.com/ggml-org/llama.cpp/pull/30168 ; https://github.com/vllm-project/vllm .
