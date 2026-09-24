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

1. CORRECTED per GPT review: `ggml_backend_cuda_buffer_type_alloc_buffer(buft, size)` (ggml-cuda.cu:883) is the WRONG shadow-creation hook -- it has no tensor/name/type argument and runs BEFORE weight upload, so it cannot know which tensor it is allocating for or hold post-upload data. Create shadows only AFTER the load_all_data loop completes (verified: src/llama-model.cpp:1871 calls `ml.load_all_data(...)`, and src/llama-model-loader.cpp:1493 defines `llama_model_loader::load_all_data`) -- shadow population must happen once real weight bytes are resident.
2. Add CUDA-backend shadow ownership keyed by the original `ggml_tensor *`, stored in the existing `struct ggml_backend_cuda_buffer_context` (verified real struct at ggml-cuda.cu:726) rather than inventing a new ownership structure.
3. Populate each shadow using `ggml_get_to_fp16_cuda(src->type)` (verified real function, declared convert.cuh:13, defined convert.cu:547, already used at ggml-cuda.cu:1404 and by conv2d.cu/fattn-common.cuh) -- reuse this exact API rather than writing new dequant math.
4. Define eligibility explicitly: dense (non-MoE-expert) weight tensors only, an explicit opt-in allowlist/pattern with explicit MoE-expert exclusion, never a global default.
5. Wire the DENSE MUL_MAT dispatch to look up a shadow BEFORE the existing `ggml_cuda_should_use_mmq(src0->type, cc, ne11/ne12, n_experts)` decision (verified real calls at ggml-cuda.cu:1872/1899/1940) -- a shadow hit bypasses native MMQ only for eligible dense tensors at/above the configured M threshold; a miss/ineligible tensor falls through to the existing should_use_mmq decision unchanged.
6. Add explicit shadow-buffer cleanup in `ggml_backend_cuda_buffer_context`'s destructor (currently unspecified) so shadow memory is freed with its owning buffer.
7. Validate shadow contents against `ggml_get_to_fp16_cuda`'s own reference output (bit/tolerance match).
8. Measure model-load overhead and VRAM delta for Qwen3.6-27B Q8_0 primary, Q4/Q6 economics controls.
9. Measure PP across M/ubatch 64..4096 and confirm TG/decode-only neutrality (shadow must not regress small-batch decode).
10. Selected-tensor-shadow vs all-eligible-shadow A/B; expose explicit opt-in flag; publish durable candidate identity so PRBE30/PRBE31 can depend on it.

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

2026-09-24 GPT review req_2b717df095b44703 applied: corrected the shadow-creation hook -- ggml_backend_cuda_buffer_type_alloc_buffer has no tensor identity and runs pre-upload, so it cannot be the creation site. Moved shadow creation to after llama_model_loader::load_all_data (verified real call sites in llama-model.cpp:1871/llama-model-loader.cpp:1493), specified ownership in the existing ggml_backend_cuda_buffer_context struct (ggml-cuda.cu:726), specified the exact reusable conversion API ggml_get_to_fp16_cuda (convert.cuh:13/convert.cu:547, verified in use elsewhere), the exact lookup point relative to ggml_cuda_should_use_mmq (verified real calls at ggml-cuda.cu:1872/1899/1940), and added explicit dense-weight allowlist/expert-exclusion/cleanup requirements that were previously unspecified.

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
- 2026-09-24T04:46:40.269201+00:00 (updated-by): Updated: section:steps, section:notes
