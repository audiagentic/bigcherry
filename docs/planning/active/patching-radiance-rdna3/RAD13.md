---
id: RAD13
order: 0
plan: patching-radiance-rdna3
state: pending
created-at: '2026-10-19T12:23:26.909976+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: L
---

# FP8 block-scaled expert path on gfx1100 for the moe worker (Qwen3.6-35B-A3B)

## Description

The moe worker is shape-generic (RCS09, PRs #317-#328): the gate is now the backend weight-format probe. gfx1100 lists only W4NL64, so Qwen3.6-35B-A3B FP8 (256 experts, hidden 2048, expert width 512, E4M3 weights with 128x128 bf16 block scales) is refused by name. This item adds the FP8 path. Design from GPT review 2026-10-19: FP8 W8A16 with BF16 activations first, preserving the checkpoint representation (codes uint8[N][K] E4M3FN, scales bf16[ceil(N/128)][ceil(K/128)]), FP16 WMMA for prefill and a separate decode GEMV; do not resurrect the shelved r4d_gemm_fp8a8 rewrite first (radiance 1.8 changed operand ownership, 128x128 prefill tiling, narrow path and epilogue, so its eight rewrite blocks no longer match).

## Steps

1. Backend entry and layout (S): gfx1100 lists MOE_FMT_FP8_E4M3_BLOCK with MOE_ACT_BF16 and bf16 128x128 scales; plan_weight/upload_weight for the FP8 layout (no W4 nibble packing, LUT or per-64 scales); both parity classes the same width; probe, byte sizing and upload round-trip tests without a model load.
2. GPU numerical harness (M): extend --route-selftest with synthetic experts from moe_fp8_ref.h run through the worker lane on gfx1100 against the host reference (GU 1024x2048 and DN 2048x512, adjacent-tile scales, zero/subnormal/negative/max E4M3, GEMM/GEMV threshold); BF16-rounded reference inputs; provisional tolerances atol/rtol 0.02 per GEMM and 0.05 per expert forward, calibrated on the synthetic distribution.
3. FP8 prefill GEMM (L): the existing moe_ plugin call is W4-specific (RAD_F8E4M3 activations, F32 group scales, W4 tables), so a new FP8 plugin dispatch variant is needed: first predecode FP8 to scaled FP16 at upload (2x memory) to validate numerics with a conventional FP16 WMMA grouped GEMM, then FP8-resident weights decoded in the GEMM (LDS staging, scale reuse across 16x16 tiles).
4. FP8 decode GEMV (M): lane_moe_gemv_fp8 beside lane_moe_gemv; fp32 accumulate over decode_e4m3(code) * scale * activation; no WMMA for one-row decode.
5. Performance and integration (M): benchmark decode and prefill separately; then Qwen3.6-35B-A3B end to end through the worker with the R9700 engine and the XTX workers.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Flash-Next W4 replay stays bitwise identical at every step; FP8 GPU output agrees with moe_fp8_ref within the recorded tolerances; probe accepts FP8 only once steps 1-4 pass; Qwen3.6-35B-A3B end to end: correct output and acceptance, then multi-stream stream-bench.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Unverified by the review: an existing compatible FP16 grouped-GEMM plugin registration and the exact plugin tensor order; inventory the native kernels before step 3. Pitfalls on gfx1100: no FP8 matrix instructions, E4M3 decode and scale multiply can dominate short GEMMs, LDS expansion halves tile capacity, FP16 overflow when scaling before WMMA, scale indexing must follow logical N/K after any relayout. Related: RCS01, RCS05, RCS09, RAD11.

## Change Log

- 2026-10-19T12:23:26.909976+00:00 (created-by): Created by agent
- 2026-10-19T12:23:41.009876+00:00 (updated-by): Updated: section:description, section:steps, section:validation, section:notes
