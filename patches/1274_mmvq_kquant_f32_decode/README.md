# 1274: dense Q6_K single-column decode without activation quantization

## Scope

Extends 1241's `f32_act` MMVQ path to Q6_K. Q6_K weights are dequantized and dotted against the original F32 activation, skipping Q8_1 activation quantization. Eligibility matches 1241: dense/non-MoE Q6_K, `ne1 == 1`, gfx1100/RDNA3_0, no forced autotune candidate. Requires 1241. An earlier Q4_K arm was removed (no gain).

## Numerics

Arithmetic changes from Q8_1-quantized activation to F32 activation/F32 accumulation, so stock bit parity is not the oracle. Correctness is test-backend-ops Q6_K MUL_MAT within CPU-reference tolerance on both arms. MTP verify (`ne1>1`) stays on the stock path.

## Hardware evidence (exploratory, pre-contract)

Dual gfx1100, 27B UD-Q6_K: decode +1.6..+1.7%, prefill -0.43% (order-balanced A/B, not contract-admitted).

## Contract

`RD74-MMVQ-Q6_K-F32-DECODE`: positive tierB-qwen9b-q6k plain decode on one gfx1100; control tierL-qwen27b-q8 (no Q6_K weights) on both cards with `-sm tensor`; activation marker `BIGCHERRY_PATCH_HIT patch=1274_kquant_f32 path=f32_decode type=q6_k ncols=1` on subject only, never ncols 2..8; 4 sessions x 10 paired rounds.

## Promotion scope

gfx1100 dense Q6_K plain non-speculative decode only. Other architectures, quant types, MoE/ID matmuls and wider batches are out of scope.
