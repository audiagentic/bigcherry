# 1309_rms_norm_mul_q81

**Status:** validated
**Plan item:** PRBE06/QFP13

## What it does

With `BIGCHERRY_RMS_Q81=1` and `GGML_HIP_Q8_1_CACHE_MODE=on` (1307), a decode-shaped fused RMS_NORM+MUL (ncols >= 1024 and a multiple of 32,
<= 16 rows, contiguous output) runs `rms_norm_mul_q81_f32`, which writes the normal F32 result and, in the same
launch, the Q8_1 blocks of that result into a 1235 cache slab published under the exact key
`ggml_cuda_mul_mat_vec_q` builds for src1 == the MUL node. Its MMVQ consumer then hits the cache and skips the
standalone `quantize_q8_1` launch. Q8_1 layout and math match `quantize_row_q8_1_cuda` (padded rows, zero padding
blocks, amax/127, roundf, half2(d, sum)), so the result is bit-identical. PRBE06 Stage 0 measured 29.9 such
RMSNorm -> quantize -> MMVQ triples per generated token per GPU with 1307 + 1308 on. Any reservation failure (e.g.
during graph capture) or ineligible shape uses the unchanged kernel. Activation evidence:
`BIGCHERRY_PATCH_HIT patch=1309_rms_norm_mul_q81` under `BIGCHERRY_PATCH_TRACE`.

## Hardware result (2026-10-04 review)

Profile v3 adoption ABBA (with 1307/1308/1310): +3% ~10K, +3.6% ~80K, greedy identical.
