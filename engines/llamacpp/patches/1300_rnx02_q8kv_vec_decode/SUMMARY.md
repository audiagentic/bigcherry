# 1300_rnx02_q8kv_vec_decode

**Status:** untested
**Plan item:** RNX02
**Design review:** GPT gateway request `req_83e7cdc000be4b9e`

## What it does

Adds a default-off selector experiment in `ggml_cuda_get_best_fattn_kernel()`.
With `BIGCHERRY_RNX02_Q8_VEC=1`, the selector chooses the existing native Q8_0
vector FlashAttention implementation only for D=256, single-token Q, Q8_0 K/V,
an available D=256 vector case, and RDNA3/RDNA4.

## Scope

This package adds no kernel and no dequantizer. It leaves the current selector
unchanged unless the exact opt-in predicate matches, and leaves gfx103x on the
existing path. The experiment must be measured against backend-reference and
unprofiled end-to-end decode before any lifecycle promotion.

## Activation evidence

Set `BIGCHERRY_PATCH_TRACE=1` alongside the opt-in flag to emit a one-time
`BIGCHERRY_PATCH_HIT` marker when the dispatch branch is selected.

## 2026-10-19 RNX02/BCOP325 reconciliation

Upstream [llama.cpp #30803](https://github.com/ggml-org/llama.cpp/pull/30803) adds a *different* gfx1201 Q4_0/Q8_0 block-per-lane VEC GQA-sharing kernel. **1300 does not implement it**: 1300 remains a default-off Q8_0/D256/Q=1 selector override only. Do not broaden this package, force Q4_0 into old VEC, or reuse the rejected 1298 multi-query result as a GQA-kernel benchmark. RNX02 owns route-first and upstream-pin admission; no patch or hardware job is authorized before the physical gfx1201 Q4_0 K/V critical-path gate. See BCOP325.

