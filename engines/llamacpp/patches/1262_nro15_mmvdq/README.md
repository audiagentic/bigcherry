# 1262 (NRO15 / PNRO15): dequant-float matvec (mmvdq) for Q4_K/Q5_K/Q6_K

## Scope

Ports nasone commit `670512936d83d12dd41d27c8b30e4bb416a47f31` (AMD). For
single-token (`ne11 == 1`), contiguous, non-batched Q4_K/Q5_K/Q6_K matvec,
`ggml_cuda_mul_mat` routes to a kernel that dequantizes the weights to float
and dots them against the F32 activation, skipping the Q8_1 activation
quantize that MMVQ needs. The fork's `mmvdq.cu`/`mmvdq.cuh` are folded into
`mmvq.cu`/`mmvq.cuh` (the patch engine edits existing files only).

Excluded from this package: the fused gate+up SwiGLU route (compiled, never
routed) and the commit's RDNA3.5 graph-opt default (PRBE36/1217 scope).

## Selection

| Control | Default | Effect |
|---|---|---|
| `GGML_CUDA_DQ_MMV` | on for RDNA3.5, off elsewhere | 0 = off, non-zero = on |
| `GGML_CUDA_DQ_Q6K` | as above | gates the Q6_K path separately |
| `GGML_CUDA_DQ_ROWS` | 1 | rows per block (1/2/4/8) |

On gfx1100/gfx1201 the path is opt-in; validation enables it on both arms.

## Validation

Contract `NRO15-MMVDQ-KQUANT-DECODE` (owner policy, 4 sessions per
architecture), producer `nro15`:

- correctness: test-backend-ops `MUL_MAT` for `type_a` in {q4_K, q5_K, q6_K}
  within the CPU-reference tolerance on both arms (float dot vs Q8_1 dot is
  not bit-identical);
- activation: `BIGCHERRY_PATCH_HIT patch=1262_nro15 path=mmvdq` on the
  subject only;
- performance: tg128 decode on `tierA-qwen4b-q6k` (positive), pp512 prefill
  (control, never routed).

```bash
bash tools/lab/plan-qualification/run_campaign.sh 1262_nro15_mmvdq \
  1262_nro15_mmvdq/nro15 gfx1100 0 <run-name>
```

## Status

untested.
