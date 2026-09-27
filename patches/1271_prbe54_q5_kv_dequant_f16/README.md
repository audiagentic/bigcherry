# 1271 PRBE54 Q5 KV F16 dequant

Untested HIP FlashAttention experiment for legacy Q5_0/Q5_1 KV cache.

## Scope

`convert.cu` adds a 32-thread kernel that dequantizes two Q5 blocks per workgroup with half2 arithmetic and an FA-specific selector. `convert.cuh` exposes that selector. The contiguous K and V F16-staging selector calls in `fattn-common.cuh` are routed through it. All non-FA callers continue to use `ggml_get_to_fp16_cuda()` unchanged. Q4_0/Q4_1 are excluded because b11126 already has dedicated converter kernels; the direct vector-FA Q5 dequant helpers are also already specialized, so 1271 targets only the `need_f16_K`/`need_f16_V` staging fallback.

Activation marker:
`BIGCHERRY_PATCH_HIT patch=1271_prbe54_q5_kv_dequant_f16 path=q5_fattn_f16_dequant contract=PRBE54-Q5-KV-DEQUANT-F16`

## Validation

Contract `PRBE54-Q5-KV-DEQUANT-F16`: gfx1100/gfx1201, full-vocabulary backend-reference correctness with Q5_0 KV + FlashAttention, Q5_0 decode positive lane, F16 prefill control, `improvement_no_regression_v1`, 4 sessions, 10 paired rounds. State remains `untested` until fresh hardware evidence exists.
