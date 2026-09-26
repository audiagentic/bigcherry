# 1262_nro15_mmvdq

**Status:** rejected
**Plan item:** NRO15

## What it does

Adds a dequant-float matvec (mmvdq) for Q4_K/Q5_K/Q6_K single-token decode: weights are dequantized to float and dotted against the F32 activation, skipping the Q8_1 activation quantize that mmvq needs. Default on for RDNA3.5 only; `GGML_CUDA_DQ_MMV=1` enables it elsewhere (`GGML_CUDA_DQ_Q6K`, `GGML_CUDA_DQ_ROWS` refine it).

## Why

The Q8_1 quantize launch is a fixed per-matvec cost in decode; for bandwidth-bound K-quant decode the float path can be cheaper.

## Upstream

Port of nasone commit `670512936d83d12dd41d27c8b30e4bb416a47f31` (AMD). The kernel files are folded into mmvq.cu/.cuh; the fused SwiGLU route and the RDNA3.5 graph-opt default are not ported.

## Rejection (2026-09-26, owner-approved)

Measured on the hardware we own it is a regression or neutral, never a win:

- gfx1201 (RDNA4): consistent -3.8 to -4.3% end to end (4-session FAIL). PVPS10 decode
  profile: `mul_mat_vec_dq_q6_K` ~57 us/call vs ~35 us/call for the `mul_mat_vec_q` calls it
  replaces; total GPU kernel time +4.6% (3908.0 -> 4089.5 ms).
- gfx1100 (RDNA3): 4-session FAIL. Profile: ~23.6 vs ~24.4 us/call, kernel time -0.9%
  (mostly the skipped `quantize_q8_1`), invisible end to end at 13% GPU-busy decode.
- Its default-on target, RDNA3.5 (gfx115x), is not available to validate; campaigns forced it
  on with `GGML_CUDA_DQ_MMV=1`. Owner decision: reject rather than hold for RDNA3.5.

