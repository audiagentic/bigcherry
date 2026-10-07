# 1347_f32_thin_transposed_mmvf

**Status:** validated
**Plan item:** QFP34

## What it does

An F32 matmul whose weight has 2 to 8 rows and whose activation has more than 8 columns runs through the float vector
kernel (`mul_mat_vec_f`) with the roles swapped, instead of rocBLAS SGEMM: the activation matrix is the matrix, the
weight rows are the vectors. One launch computes the result transposed into a pool buffer and a small kernel writes it
into the destination. On by default; `BIGCHERRY_F32_THIN_MMVF=0` restores SGEMM.

## Why

Flash-Next's hyper-connection blocks project the 10240-wide state onto 4 values per token (`hc_attn_inject`,
`hc_ffn_inject`, F32 [10240, 4]), twice per layer: 96 calls per 512-token prefill chunk, each a GEMM with a 4-row
weight. SGEMM's tiled kernel is built for large square problems. In the prefill kernel profile of the production
build (run `gate0-d24576`) the SGEMM kernel serving these calls and the router is 12.7% of an XTX's kernel time, at
332 us per call. Upstream already does the role swap for a one-row weight; this is the same for the batch width the
vector kernel supports.

## Scope

Plain MUL_MAT only (not MUL_MAT_ID), F32 weight and activation, contiguous tensors, no batch dimensions. Wider
weights (the 512-row router, the 48-row SSM projections) stay on SGEMM. The sums are formed in another order than
SGEMM forms them: equal within float tolerance, not bit-identical.

## Activation

`BIGCHERRY_PATCH_TRACE=1` prints `BIGCHERRY_PATCH_HIT patch=1347_f32_thin_transposed_mmvf k= rows= cols=` once.
