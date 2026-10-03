# 1310_act_q81

**Status:** untested
**Plan item:** QFP13

## What it does

With `BIGCHERRY_ACT_Q81=1` and `GGML_HIP_Q8_1_CACHE_MODE=on` (1307), an F32 plain (`ggml_cuda_op_unary`) or gated
(`ggml_cuda_op_unary_gated`: SWIGLU/GEGLU/...) activation whose row length is a multiple of 32 (rows are written in MMVQ's padded layout,
GGML_PAD(ne0, 512) with zero padding blocks), with <= 16 rows and a contiguous output (routed-expert activations are excluded: their MoE MMVQ consumers do not hit the published key yet), runs `bc_act_q81_kernel`: it writes the normal F32
result and, in the same launch, native-layout Q8_1 blocks into a 1235 cache slab published under the key
`ggml_cuda_mul_mat_vec_q` builds for src1 == this node. Its MMVQ consumer then skips the standalone `quantize_q8_1`
launch (QFP13 census: ~30 gated + ~30 plain activation -> quantize -> MMVQ chains per generated token per GPU).
The Q8_1 math matches `quantize_q8_1` exactly. Ineligible shapes or a failed reservation run the unchanged kernel.
Activation evidence: `BIGCHERRY_PATCH_HIT patch=1310_act_q81` under `BIGCHERRY_PATCH_TRACE`.
