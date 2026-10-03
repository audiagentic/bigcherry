---
id: RNX05
order: 5
plan: patching-r9700-cross-rdna
state: pending
created-at: '2026-10-03T01:33:08.311145+00:00'
breadth: ''
skill: advanced
created-by: codex
work: L
priority: P3
---

# R9X05 — MXFP4/FP8 grouped MoE and skinny GEMM extraction

## Description

Determine whether OCP-MXFP4-weight × FP8-activation grouped MoE and skinny FP8 GEMM concepts have a deployable llama.cpp quantized model path. Keep format work separate from ordinary MMQ tile tuning.

## Steps

- Review 1237, 1273, and 1274 plus current quant type/conversion code before inventing a GGML type.
- Answer whether GGUF can represent weights/scales, conversion is offline/load-time, FP8 activation is usable per target, and bandwidth reduction offsets conversion/portability cost.
- Implement 1305 only with real type/storage and expert-map integration; implement 1306 only after shared format helpers exist and traces show a separate skinny hotspot.
- Retain existing compact-grid ownership and provenance/attribution for adapted source.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files

- kernels/r9k_moe_mxfp4a8.hip
- kernels/r9k_gemm_fp8.hip
- r9700_vllm/kernels/moe.py
- tools/gen_kmag.py
- tests/test_fold_mxfp4.py
- ggml/src/ggml-cuda/mmq.cuh
- ggml/src/ggml-cuda/mmvq.cu

## Validation

Bit/ULP unpack, scales, padding, expert-boundary and output tests; hostile routing; M crossover; 1/2/3-GPU split; conversion/load cost, VRAM bytes, kernel/E2E speed, and source-license provenance.

## Effort & Risk



## Standards

RDNA4 is reference; RDNA3 distinguishes algorithm portability from native datatype support; RDNA2 uses existing Q4/Q8 or FP16/BF16 concepts.

## Acceptance Criteria

- Acceptance requires a real GGUF/model path and repeatable E2E win.
- If native format is not deployable, port only layout/epilogue ideas into existing Q/IQ kernels and close the format-specific experiment.
- Do not software-emulate FP8 for its own sake on unsupported generations.

## Notes

Original source alias is R9X05. Existing owners: 1203, 1208, 1237, 1241, 1245, 1262, 1265, 1267, 1273, 1274. Proposed slots 1305 and 1306.

Verbatim legacy source retained during R9X→RNX migration:

# R9X05 — MXFP4/FP8 grouped MoE and skinny GEMM extraction

Status: planned
Proposed patches: `1305_r9x_mxfp4_moe_gemm`, `1306_r9x_fp8_skinny_gemm`
Depends on: R9X01
External source: `kernels/r9k_moe_mxfp4a8.hip`, `kernels/r9k_gemm_fp8.hip`, `r9700_vllm/kernels/moe.py`, `tools/gen_kmag.py`, `tests/test_fold_mxfp4.py`
Existing BigCherry owners: 1203, 1208, 1237, 1241, 1245, 1262, 1265, 1267, 1273, 1274

## Goal

Determine whether r9700-stack's OCP-MXFP4-weight × FP8-activation grouped MoE and skinny FP8 GEMM concepts can become useful llama.cpp quantized kernels, and which parts can be generalized to RDNA3/RDNA2. This is format work, so it must not be mixed with ordinary MMQ tile tuning.

## Exact BigCherry target family

Review `patches/1237_rd30_moe_mmq_compact_grid/patch.py` first; it modifies `ggml/src/ggml-cuda/mmq.cuh` and BigCherry MMQ workspace accounting. Review `patches/1273_iq_mmvq_rdna_tuning/patch.py`, whose target includes `ggml/src/ggml-cuda/mmvq.cu` plus quant vec-dot definitions, and `1274_mmvq_kquant_f32_decode`. Also inspect current quant type definitions/conversion code before inventing a GGML type.

1305 target set is expected to include the current MMQ/MUL_MAT_ID dispatch (`ggml/src/ggml-cuda/mmq.cuh` and its launch unit) plus quantization/type plumbing only if the model file can actually carry MXFP4/E8M0 data. R9X01 must resolve exact type/storage path first. 1306 should target the existing dense skinny GEMM/MMVQ dispatcher instead of duplicating 1305's datatype plumbing.

## Decision gate before code

Answer: (a) can GGUF represent the source weight/scales exactly or via a clean new type; (b) can conversion happen offline/load-time rather than every inference; (c) does llama.cpp have FP8 activation support usable on each target arch; (d) is the expected bandwidth reduction large enough to offset conversion and lower portability. If any answer is no, port the layout/epilogue ideas into existing Q/IQ kernels instead and close the format-specific patch.

## Implementation split

`1305`: grouped expert kernel, expert-map/bounds integration, folded unpack/layout, activation quantization, correctness against dequantized reference. Keep `1237/1265` compact-grid ownership intact.

`1306`: only after 1305's shared format/helpers exist; implement dense/skinny M small path (LM head/router-like shapes) if traces show a separate hotspot. Do not duplicate helpers.

## RDNA adaptation

RDNA4: native FP8/MXFP4 path is the reference. RDNA3: distinguish algorithm portability from native datatype support; if gfx11 cannot execute the same low-precision matrix op efficiently, reuse packed layout/activation quantization only if an MFMA/vector unpack path wins E2E. RDNA2: treat native FP8 assumptions as unsupported; test a concept port into existing Q4/Q8 or FP16/BF16 kernels rather than software-emulating FP8 for its own sake.

## Validation

Bit/ULP tests for unpack, scales, padding rows, expert boundaries and output; hostile routing distributions; all M crossover points; 1/2/3-GPU split. Measure conversion/load cost and VRAM bytes as well as kernel/E2E speed. Validate source license/provenance: retain required attribution for adapted source and do not silently copy third-party code with different provenance.

Acceptance requires a real GGUF/model path and E2E win. A synthetic FP8 GEMM win without a deployable storage/conversion path is insufficient.



## Change Log

- 2026-10-03T01:33:08.311145+00:00 (created-by): Created by codex
- 2026-10-03T01:38:49.432431+00:00 (updated-by): Updated: section:notes
