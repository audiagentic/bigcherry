---
id: PRBE29
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:55:27.064093+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: null
---

# AMD-GEMM-002: Persistent F16 shadow of quantized dense weights

## Description

TODO, first-principles investigation, foundational for PRBE30/PRBE31. Persistent F16 device-buffer 'shadow' of selected quantized dense (non-MoE-expert) weights, built once at model load (not per-GEMM), opt-in only. Real b11126 confirmed this batch: ggml-cuda.cu has cuBLAS integration (`ggml_cuda_mul_mat_cublas_impl` at line 1433, `cublasHandle_t`/`CUBLAS_CHECK` throughout) but NO hipBLASLt-specific path exists yet -- so this project's HIP build currently runs dense F16/F32 GEMM through hipBLAS's cuBLAS-compatibility shim, not a tuned hipBLASLt path; introducing the latter (needed for PRBE31's crossover measurement) is itself new integration work, not merely a dispatch flag.

## Steps

1. Identify the model-load-time device-buffer allocation path (ggml-cuda.cu:883 `ggml_backend_cuda_buffer_type_alloc_buffer`, confirmed this batch) and the point after weight upload where a one-time dequant-to-F16 pass could run for SELECTED eligible dense tensors only.
2. Define eligibility: dense (non-MoE-expert) weight tensors only, explicit opt-in list/pattern (never a global default, never MoE experts by default per item text).
3. Implement the persistent F16 shadow allocator: one dequant kernel launch per eligible tensor at load time, populate shadow buffer, keep original quantized buffer resident (shadow is additive, not a replacement, since small-batch decode should keep using native MMQ on the quantized buffer).
4. Validate shadow contents against a dequant reference (bit/tolerance match).
5. Measure model-load overhead (extra time) and VRAM delta (shadow size) for Qwen3.6-27B Q8_0 primary, Q4/Q6 economics controls.
6. Wire the DENSE MUL_MAT dispatch (ggml_cuda_mul_mat_cublas_impl or the MMQ/cuBLAS selection point around it) to optionally use the F16 shadow via existing cuBLAS (not yet hipBLASLt -- that is PRBE31's scope) for M/ubatch >= some threshold, native MMQ below it.
7. Measure PP across M/ubatch 64..4096 and confirm TG/decode-only neutrality (shadow must not regress small-batch decode).
8. Selected-tensor-shadow vs all-eligible-shadow A/B; expose explicit opt-in flag; publish durable candidate identity so PRBE30/PRBE31 can depend on it.

## Detailed Solution & Technical Design

This is new, first-principles work spanning model-load code and the dense GEMM dispatch point. The dequant-to-F16 kernel itself can likely reuse existing per-type dequantize_* device functions already in the codebase (used by MMVQ's own dequant paths) rather than writing new dequant math -- implementer should grep ggml-cuda/dequantize.cu or per-type convert.cu (convert.cu confirmed to exist at b11126) before writing a new dequant kernel. The dispatch-threshold logic (native MMQ vs shadow+GEMM) is new decision code at the ggml_cuda_mul_mat top-level dispatcher.

## Code Samples & Guidance

Real b11126 anchors (verified): ggml-cuda.cu:883 buffer allocation; ggml-cuda.cu:1433 `ggml_cuda_mul_mat_cublas_impl(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, ...)` existing cuBLAS GEMM path (the shadow's eventual consumer, via cuBLAS for now, hipBLASLt in PRBE31); ggml/src/ggml-cuda/convert.cu/.cuh (existing dequant kernels, likely reusable for the shadow-population kernel -- exact function names not read this batch, must be verified). No hipBLASLt symbols found anywhere in ggml-cuda.cu at b11126 (grepped this batch) -- flag explicitly: PRBE29 itself only needs cuBLAS (already present) since its own PP measurement can run through the existing cuBLAS path; hipBLASLt integration is exclusively PRBE31's new-work scope.

## Files

ggml/src/ggml-cuda/ggml-cuda.cu (buffer alloc, dense MUL_MAT dispatch threshold, ~883 and ~1433); ggml/src/ggml-cuda/convert.cu/.cuh (reuse for shadow dequant); model-load path (llama.cpp weight-loading, opt-in flag); patches/12xx_rd36_f16_shadow_dense/ (new, base patch for the chain -- see notes on PRBE30/31 layering via `requires`).

## Validation

Shadow contents vs dequant reference; model output/PPL parity; load-time overhead; VRAM delta; PP across M/ubatch 64..4096; TG/decode-only neutrality control; selected-vs-all-eligible-tensor arms; VRAM-constrained control. Brutus hardware run not performed here.

## Effort & Risk

L: spans model-load code, a new dequant-at-load kernel invocation, and dense-dispatch threshold logic; foundational for PRBE30/31 so correctness bugs here propagate downstream.

## Standards

Selective shadow; explicit resource accounting; no global default; dependency-aware promotion.

## Acceptance Criteria

A declared target workload shows repeatable PP gain that justifies VRAM/load cost; no global default and no MoE shadow explosion; otherwise retain negative evidence and leave dependents blocked.

## Notes

Supersedes: RD36
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd36

2026-09-24 relevance at b11126: TODO, foundational. Confirmed NO hipBLASLt integration exists at b11126 (only cuBLAS/hipBLAS-compat) -- PRBE29 itself can and should measure through the existing cuBLAS path; PRBE31 alone needs new hipBLASLt integration. GPT design request for PRBE29+30+31 hit a queue-saturated gateway and was not obtained in-session; plan authored directly from verified source, dequant-kernel-reuse anchor flagged as unverified.

## Change Log

- 2026-09-09T10:55:27.064093+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:12:39.409528+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.258047+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.995120+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:56:51.645822+00:00 (updated-by): Updated: section:description, section:steps, section:files, section:validation, section:standards, section:acceptance_criteria
- chg_20260910_025719_dense-gemm-successors-prbe293_6872
- 2026-09-10T02:57:19.678291+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T02:33:57.527150+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:effort_risk, section:notes
