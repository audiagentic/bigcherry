# 1265_rd30b_moe_mmq_compact_grid_rdna4_rdna2

Plan `RD30`; requires `1237_rd30_moe_mmq_compact_grid` (promoted 2026-09-26, in `validated-enhancements`, so the scaffold control already carries it gated off on these cards).

One edit to `ggml/src/ggml-cuda/mmq.cuh`: `ggml_cuda_mmq_moe_compact_enabled()` returns true for `cc == GGML_CUDA_CC_RDNA3 || GGML_CUDA_CC_IS_RDNA4(cc) || GGML_CUDA_CC_IS_RDNA2(cc)`. RDNA3.5 stays excluded (no hardware).

Validation (contract `RD30B-MOE-MMQ-COMPACT-GRID-RDNA4-RDNA2`, producer `rd30b`, copied from 1237's): byte-identical 256-expert MUL_MAT_ID test-backend-ops output (Q4_K, Q8_0, 3 seeds), the 1237 compact-grid marker on the subject only, pp512 positive and tg128 control lanes on Qwen3.6-35B-A3B, four sessions each on gfx1201 (device 2) and gfx1030 (device 3).
