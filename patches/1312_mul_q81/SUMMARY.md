# 1312_mul_q81

**Status:** untested
**Plan item:** QFP13

## What it does

With `BIGCHERRY_ACT_Q81=1` and `GGML_HIP_Q8_1_CACHE_MODE=on` (1307), a same-shape (no broadcast), contiguous F32
`GGML_OP_MUL` with a row length that is a multiple of 32, in a decode-shaped graph, runs `bc_mul_q81_kernel`: it writes
the normal F32 product and, in the same launch, MMVQ-padded native Q8_1 blocks published under the key its MMVQ
consumer looks up. Targets the remaining Flash-Next misses fed by a MUL: the full-attention output gate
(`attn_gated` -> wo) and the GatedDeltaNet gated output norm (`final_output` -> ssm_out). Q8_1 math matches
`quantize_q8_1` exactly; anything else runs the unchanged bin_bcast path.
Activation evidence: `BIGCHERRY_PATCH_HIT patch=1312_mul_q81` under `BIGCHERRY_PATCH_TRACE`;
`BIGCHERRY_Q81 publish-mul` under `BIGCHERRY_Q81_TRACE`.
