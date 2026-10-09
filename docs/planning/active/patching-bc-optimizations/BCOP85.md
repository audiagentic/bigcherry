---
id: BCOP85
order: 85
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-09T06:12:24+00:00'
created-by: agent
priority: P1
work: S
---

# PRBE70: fail-closed typed CK GEMM oracle instead of rejected op timing

## Discovery / disposition

PRBE70's proposed `GGML_CUDA_OP_TIMING` census exists only in rejected patch 1203 and logs untyped `MUL_MAT` `[KxMxN]`, not a device/route/stride/quantized or `MUL_MAT_ID` signature. Production already has rocprofv3 per-device kernel timing and a Flash-Next family summarizer, but neither proves tensor shape or E2E critical-path share. CK's maintained `ROCm/rocm-libraries` profiler requires explicit dtype/layout/strides and vectorized arguments for grouped GEMM. Treat quantized GGUF MMQ versus float CK as **not directly comparable**; no new CK performance measurement exists.

## Authoritative ownership / activity boundary

PRBE70 owns only offline CK eligibility and disposition. PRBE28/29/30/31 retain float padding, quantized F16 shadow, shadow padding and BLAS crossover; PKC05/HIP tuning retain MMQ/MMVQ. QFP36/41, Meta split cache, QFP35, MTP, engine migration, Radiance and patch-promote work were independently active within 12 hours and were not edited. Last independent PRBE70 plan change: 2026-10-08 06:47 UTC, outside the window. BCOP75-84 have different mechanisms; no CK oracle scheduled audit repeat.

## Bounded next action / terminal gate

Check an installed gfx1100/gfx1201 CK binary/instance and exact CLI; otherwise record `CK_UNAVAILABLE`. Reuse existing rocprof traces to select **one** actual float BLAS GEMM, bind typed tensor/device/stride/route metadata from a pinned fixture, then run CK's own verify and an identical native kernel-only control. Reject missing identity, quantized source, unmatched MoE routing and unproven critical-path attribution. Close if the theoretical E2E throughput ceiling is <3%, CK loses, or winning tiles cannot be reproduced without new runtime infrastructure. Any positive result hands a narrow default-off candidate to its existing owner; only subsequent four-session >=10 ABBA-pair E2E/correctness evidence can promote. Disposable 13/13 host admission/argv checks passed; no CK, build, GPU or hardware A/B ran.

## References

PRBE70; PRBE28/29/30/31; PKC05; rejected 1203; tools/bigcherry/profiling/rocprof.py; tools/lab/flash-next/prefill-kernel-table.py; pinned llama.cpp b11474 ggml-cuda.cu; ROCm/rocm-libraries composablekernel profiler src/profile_gemm.cpp, profile_grouped_gemm.cpp and profiler/README.md; llama.cpp PR #30168; vLLM ROCm AITER CK alignment.
