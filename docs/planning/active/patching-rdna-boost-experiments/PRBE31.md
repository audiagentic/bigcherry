---
id: PRBE31
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:55:35.677451+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: null
---

# AMD-GEMM-004: Large-M F16 shadow to tuned hipBLASLt crossover

## Description

TODO, depends on PRBE29 (+ optionally PRBE30). Determine the real M-size crossover between native quantized MMQ and tuned hipBLASLt-over-F16-shadow. This is the ONE item in the GEMM-shadow chain that needs genuinely new integration work beyond PRBE29's allocator: confirmed this batch that NO hipBLASLt symbols exist anywhere in ggml-cuda.cu at b11126 (only cuBLAS/hipBLAS-compat via `ggml_cuda_mul_mat_cublas_impl`) -- 'tuned hipBLASLt' dispatch does not exist in this codebase yet and must be added, not merely enabled.

## Steps

1. Hard-require PRBE29's F16 shadow (and optionally PRBE30's K-padding) to exist first -- CORRECTED: make this a genuinely SEPARATE package requiring PRBE29 (same self-dependency fix as PRBE30 -- do not anchor as additive edits into PRBE29's own package while also declaring requires on it).
2. Add HIP build/link wiring for hipBLASLt (currently entirely missing, per GPT review): edit ggml/src/ggml-hip/CMakeLists.txt beside the existing `find_package(hipblas REQUIRED)` / `find_package(rocblas REQUIRED)` to add `find_package(hipblaslt REQUIRED)`, and beside the final HIP libraries linked, add `roc::hipblaslt` -- verify these exact anchor lines in CMakeLists.txt at implementation time before writing the Edit().
3. Add a hipBLASLt-specific GEMM path alongside the existing `ggml_cuda_mul_mat_cublas_impl` cuBLAS path in ggml-cuda.cu, gated to HIP builds: a concrete `hipblasLtHandle_t` create/destroy lifecycle alongside the existing cuBLAS handle lifecycle (~718-741), a concrete hipBLASLt GEMM helper function, and dispatch from `ggml_cuda_mul_mat` before the native-MMQ fallback, activated ONLY when a shadow exists (PRBE29) and M crosses the configured threshold.
4. Define Qwen3.6-27B Q8_0 primary dense signatures, Q4/Q6 secondary.
5. Sweep M=64,128,192,256,384,512,768,1024,2048,4096 comparing three arms: native quantized MMQ, default (untuned) hipBLASLt-over-shadow, tuned hipBLASLt-over-shadow.
6. Validate tolerant/exact output and PPL for all three arms BEFORE any timing comparison.
7. Record per-architecture/per-quant kernel and E2E PP, VRAM, and load cost; derive the crossover point from measurement, not an assumed fixed threshold.
8. Include M<=128, decode, and memory-constrained controls.
9. If tuned native MMQ always wins, explicitly reject shadow dispatch as a promoted path while retaining EC10-style negative evidence.

## Detailed Solution & Technical Design

The item's original framing (assume hipBLASLt already exists as a callable library and this item just finds the crossover) does not hold at b11126 -- hipBLASLt integration itself is new work. This materially raises this item's effort above what PRBE30's sibling items need, since it involves adding a new GEMM backend path (handle creation/destruction mirroring the existing cuBLAS handle lifecycle at ggml-cuda.cu ~718-741, algorithm selection/tuning-cache management, and a dispatch decision integrated with PRBE29's shadow-vs-native threshold logic) before any crossover sweep can even be measured.

## Code Samples & Guidance

Real b11126 anchor (verified, for the EXISTING cuBLAS path this item's new hipBLASLt path should sit alongside, not replace): ggml-cuda.cu ~718-741 (cuBLAS handle lifecycle, `cublas_handles[i][j]`, `CUBLAS_CHECK(cublasDestroy(...))`); ggml-cuda.cu:1433 `ggml_cuda_mul_mat_cublas_impl`. No hipBLASLt anchor exists yet -- this is new-file/new-function work, not an Edit() against existing hipBLASLt code. Patch package: additive Edit set on patches/12xx_rd36_f16_shadow_dense/ (PRBE29's package) with `requires` on PRBE29 (and PRBE30 if its padding is used), adding a new hipBLASLt handle/dispatch source file plus the crossover-sweep validation script.

## Files

ggml/src/ggml-cuda/ggml-cuda.cu (new hipBLASLt handle lifecycle + dispatch, alongside existing cuBLAS code ~718-741, ~1433); a new ggml-cuda/hipblaslt-mmq.cu or similar (new file, hipBLASLt integration); PRBE29's shadow allocator (dependency); M-sweep crossover validation script.

## Validation

Output/PPL correctness for all three arms (native, default hipBLASLt, tuned hipBLASLt) before timing; M-sweep 64..4096; per-architecture/quant kernel+E2E PP; VRAM/load cost; decode and memory-constrained controls; durable crossover point or explicit negative-result rejection with EC10-style evidence retained.

## Effort & Risk

L: requires adding a new GEMM backend integration (hipBLASLt) that does not exist in this codebase today, on top of depending on PRBE29 (and optionally PRBE30) -- the largest and riskiest item in this chain; flag to reviewers that this is materially bigger than PRBE30.

## Standards

Derived threshold; dependency-aware; preserve negative result; no shadow default from kernel-only timing.

## Acceptance Criteria

Only a measured per-architecture/quant crossover with positive E2E benefit promotes; if no crossover exists, explicitly reject dispatch and retain evidence.

## Notes

Supersedes: RD38
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd38

2026-09-24 relevance at b11126: TODO, blocked on PRBE29, requires NEW hipBLASLt integration not present at b11126 (confirmed via grep -- zero hipblaslt/hipBLASLt matches in ggml-cuda.cu). This changes the item's effort estimate upward from what its own text implies (which assumes hipBLASLt dispatch already exists and only the crossover needs measuring). GPT design request for PRBE29+30+31 hit a queue-saturated gateway and was not obtained in-session; plan authored directly from verified source, and this scope gap is the most important finding to flag for GPT/human review before implementation starts.

2026-09-24 GPT review req_2b717df095b44703 applied: same self-dependency correction as PRBE30 (separate package requiring PRBE29, not additive-into-PRBE29-while-requiring-it). Added the previously entirely-missing HIP build/link wiring: find_package(hipblaslt REQUIRED) and roc::hipblaslt in ggml/src/ggml-hip/CMakeLists.txt beside the existing hipblas/rocblas entries, plus a concrete handle/dispatch design gated on shadow existence and M threshold.

## Change Log

- 2026-09-09T10:55:35.677451+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:12:47.175446+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.267648+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.009190+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:57:05.654512+00:00 (updated-by): Updated: section:description, section:steps, section:files, section:validation, section:standards, section:acceptance_criteria
- chg_20260910_025719_dense-gemm-successors-prbe293_6872
- 2026-09-10T02:57:19.708858+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T02:34:27.116647+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:effort_risk, section:notes
- 2026-09-24T04:47:01.748714+00:00 (updated-by): Updated: section:steps, section:notes
