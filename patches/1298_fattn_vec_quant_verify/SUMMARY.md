# 1298_fattn_vec_quant_verify

**Status:** untested
**Plan item:** RNX02

## What it does

On GPUs without tensor cores (HIP RDNA), quantized-KV flash attention uses the vector kernel for up to
`BIGCHERRY_FA_VEC_QMAX` queries (default 4, upstream 2) instead of the tile kernel.

## Why

The tile kernel needs f16 K/V, so every call converted the whole q8_0 cache to f16 (O(n_kv), ~1 ms/token at
80K). MTP3 verify is 4 queries, so the target paid it every step. The vector kernel reads q8_0 directly.

Validation: decode ms/MTP step A/B (q8_0 target KV) at 10K/80K/160K; greedy parity without MTP (n=1 path
unchanged) and with MTP acceptance; rocprof shows no dequantize_block_q8_0_f16 in the decode window.
