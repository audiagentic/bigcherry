# IQ8R QKV CPU/GPU difference: two-level scale precision

2026-11-22. **Root cause isolated. No production precision change made.** This is a numerical defect candidate, not a GPU-architecture disparity or an accepted quality tolerance.

## Measured control

Existing source-backed native C06 `RadCtx` block-0 tests (PRs #1315, #1321) use actual IQ8R RAD2 weights, a deterministic one-row BF16 activation, and both supported video/audio head splits. In both card orders (gfx1100/gfx1100/gfx1201 and gfx1201/gfx1100/gfx1100):

- **86,400/86,400 BF16 intermediate values** matched between card placements.
- **36,864 INT8 rotated activation codes** and **144 FP32 activation scales** matched between host and GPU; gate projection matched exactly.
- The first host-vs-device difference occurred after the IQ8R QKV GEMM: relative L2 across rank/modality/split samples **0.2545–0.3096%**, not at input quantization.
- Full C06 outputs had same-split device-versus-host relative L2 **0.330–0.422%**. Neither number is an accepted model-quality tolerance.

## Source-level difference

The checkpoint stores an `affine/fwht128` encoding with `codes` INT8, `scale` U8 per (output row, 256 K), and `scale.1` BF16 per output row.

- Reference: `libref/ref_gemm.cpp:459–478` materializes `two_level[n,g] = scale_U8[n,g] * scale.1_BF16[n]` into **FP32**, then the existing quantized GEMM consumes that FP32 scale.
- gfx1201: `libr4d/r4d_fp8_layout.cpp:235–243` declares scale layout `r4d.iq8r.s.t16`, **BF16**; `:303–317` multiplies both planes and calls `rad_store_f32(..., RAD_BF16, product)` before any GEMM launch.
- gfx1100: `libhalo/halo_fp8_layout.cpp:288–294` uses the same BF16 layout tag; `:343–351` performs the same BF16 fold. This is why the two GPUs agree exactly but disagree with reference.
- Existing device GEMMs in `libr4d/r4d_gemm_fp8a8.hip` and `libr4d/r4d_gemm_8a8_prefill.h`, with their libhalo equivalents, consume the rounded scale. This precision loss is **upstream of the GPU GEMM arithmetic**.

## Independent checkpoint-backed oracle

From the 8/8/16 C06 test capture, select 11 distributed output indices for each Q/K/V bank, each rank, both video and audio: 11 × 3 banks × 3 ranks × 2 modalities = **198 output elements**.

For each sampled row, read the **original** source checkpoint `codes`, U8 `scale`, BF16 `scale.1` and BF16 bias; reuse captured INT8 activation codes and FP32 scales. Accumulate exact INT8 products in each 256-K group; compare two formulas at the group fold:

| Scale representation | Exact host matches | Exact gfx1100/gfx1201 output matches |
| --- | ---: | ---: |
| Product retained in FP32 (`float(U8) * BF16_row`) | **198 / 198** | 126 / 198 |
| Product rounded to BF16 before GEMM | 126 / 198 | **198 / 198** |

Mean absolute output difference between the two forms across samples is **0.001033**, after the actual QKV BF16 bias epilogue. The change in the selected 72 output elements is fully explained by the extra scale rounding. Several group-level accumulation/scale application orders produced the same 198 GPU outputs, so **this result does not adjudicate every future accumulation-order difference**.

The reference implementation is not a defective match target: the decoded checkpoint scale is the FP32-representable product of a U8 integer and an already BF16-representable factor. Rounding it again to BF16 is a strictly lossy intermediate representation. **Do not make libref round to BF16 to force agreement.**

## Precision-preserving correction options

The generic Radiance kernel owner **RUM15** must measure before changing any binary or layout:

1. **F32 tile-major scale plane for IQ8R only**: reuse existing native relayout and narrow/tiled GEMM templates, change the IQ8R scale layout tag/byte count and consumer scale accessor, preserving F32 `U8 × BF16`. This doubles the **scale plane**, not the entire checkpoint. Across all 1,306 IQ8R matrices in this source container there are 127,309,824 groups: BF16 plane ~242.82 MiB; F32 plane ~485.65 MiB, an **additional ~242.82 MiB if all resident simultaneously**. Real per-rank/device residency and memory admission must be measured, not inferred from this total.
2. **Retain the original factors** (U8 group coefficient and BF16 row gain) and multiply in FP32 in existing GEMM fold. This can avoid the doubled F32 scale plane, but entails modified GPU loads/instructions and source-loader layout changes. Benchmark the actual narrow and prefill kernels and refills.

Both options require the same source-managed Radiance patch discipline and validated layout-fingerprint/ABI transition. They must preserve all non-IQ8R layouts, host/RAM/SSD loading/refill, integer group-256 accumulation, and generic row/column sharding. Do not create duplicate kernel families, a second model loader, or a special LTX-only serving path.

## Required gates

- Failing source- and output-backed QKV precision test **before** production edits, then green precision test after, retaining the unmodified FP32 reference.
- gfx1100/gfx1201 native C06 issue with 1/16/64 rows, both legal splits and reversed ranks under GPU leases; real TP1 activation and higher-precision golden comparison; no guessed tolerance.
- Exact layout contract/ABI fingerprint, kernel selection and rejection of mismatched cached layouts, source extent and weight identity, interrupted stage, restart/refill, VRAM/memory admission, and same-SDK serving canary.
- Benchmarks with captures **disabled** on XTX and R9700; test prefill/decode latency and GPU/host memory transfer. A GPU scale representation that is numerically better but fails admission is **not production-ready**.

LTX02 remains open for actual TP1 quality and row coverage; RUM15 alone owns the generic scale precision correction. Production LTX TP3 remains refused.
