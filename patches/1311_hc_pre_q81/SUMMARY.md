# 1311_hc_pre_q81

**Status:** validated
**Plan item:** QFP13

## What it does

With `BIGCHERRY_HC_Q81=1` and `GGML_HIP_Q8_1_CACHE_MODE=on` (1307), an eligible hyper-connection pre-mix
(`GGML_OP_DSV4_HC_PRE`: contiguous F32 output, n_embd a multiple of 32, <= 16 tokens) runs `dsv4_hc_pre_q81_f32`,
which writes the normal F32 result and, in the same launch, native-layout Q8_1 blocks (MMVQ's padded rows, zero
padding blocks, `quantize_q8_1` math) into a 1235 cache slab published under the key `ggml_cuda_mul_mat_vec_q`
builds for src1 == this node, so its MMVQ consumer skips the standalone quantize launch. Motivation: the largest
remaining class of Q8_1 misses after 1307-1310 (~2958 of ~8056 in `flashnext-v2-q81c-trace`). Ineligible shapes or
a failed reservation launch the unchanged kernel. Activation evidence: `BIGCHERRY_PATCH_HIT patch=1311_hc_pre_q81`.

## Hardware result (2026-10-04 review)

Adopted in profile v3 2026-10-04: quantize/token 74 -> 45, kernels/token 1138 -> 1116; screens neutral (~24K) to ~-2% ms/step (~80K); greedy identical.
