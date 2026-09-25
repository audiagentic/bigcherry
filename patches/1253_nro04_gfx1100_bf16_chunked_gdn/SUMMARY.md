# 1253_nro04_gfx1100_bf16_chunked_gdn

**Status:** untested
**Plan item:** NRO04

## What it does

Adds the nasone fork's BF16/WMMA chunked GatedDeltaNet prefill kernels (gfx11 and gfx12, new files) and routes K == 1 prefill with S_v == 128 on RDNA3/RDNA4 to them by default, falling back to the sequential kernel if the driver rejects the launch. Opt out with `GGML_CUDA_GDN_CHUNKED_BF16=0`.

## Why

Sequential GDN recurrence dominates prefill on hybrid Qwen models; the chunked form uses tensor cores. Near-lossless, not bit-exact.

## Upstream

Port of nasone commit `4169fbbf50d24beb6d269a2350e7f780b85369e6` (block 02). The fp32 chunked kernel is not ported (RD50/1221 scope; conflicts with 1221).
