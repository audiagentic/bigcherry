# 1262_nro15_mmvdq

**Status:** untested
**Plan item:** NRO15

## What it does

Adds a dequant-float matvec (mmvdq) for Q4_K/Q5_K/Q6_K single-token decode: weights are dequantized to float and dotted against the F32 activation, skipping the Q8_1 activation quantize that mmvq needs. Default on for RDNA3.5 only; `GGML_CUDA_DQ_MMV=1` enables it elsewhere (`GGML_CUDA_DQ_Q6K`, `GGML_CUDA_DQ_ROWS` refine it).

## Why

The Q8_1 quantize launch is a fixed per-matvec cost in decode; for bandwidth-bound K-quant decode the float path can be cheaper.

## Upstream

Port of nasone commit `670512936d83d12dd41d27c8b30e4bb416a47f31` (AMD). The kernel files are folded into mmvq.cu/.cuh; the fused SwiGLU route and the RDNA3.5 graph-opt default are not ported.
