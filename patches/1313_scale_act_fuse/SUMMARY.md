# 1313_scale_act_fuse

**Status:** untested
**Plan item:** QFP13

## What it does

With `BIGCHERRY_SCALE_ACT_FUSE=1`, `ggml_cuda_try_fuse` runs `SCALE -> UNARY(SILU|SIGMOID)` and
`SCALE -> UNARY(SILU|SIGMOID) -> SCALE` chains (F32, contiguous, single-use intermediates) as one `bc_scale_act_kernel`
launch computing the same expressions in the same order (bit-identical). Targets the Flash-Next hyper-connection
blocks: `silu(scale(w_down @ xn))` and `2 * sigmoid(scale(inject))`, ~98 scale launches per token per GPU. When the
chain ends at the activation and 1310's rules hold (`BIGCHERRY_ACT_Q81=1`, cache on, decode graph), the launch also
publishes the MMVQ Q8_1 activation like 1310. `UNARY -> MUL` is left to upstream's unary_mul fusion (1312).
Activation evidence: `BIGCHERRY_PATCH_HIT patch=1313_scale_act_fuse` under `BIGCHERRY_PATCH_TRACE`.
