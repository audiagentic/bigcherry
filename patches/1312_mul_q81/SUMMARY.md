# 1312_mul_q81

**Status:** untested
**Plan item:** QFP13

## What it does

Upstream fuses `UNARY(sigmoid/silu/softplus) -> MUL` pairs into `ggml_cuda_op_unary_mul` (one gated launch writing the
MUL node), which bypasses 1310's producers. With `BIGCHERRY_ACT_Q81=1` and `GGML_HIP_Q8_1_CACHE_MODE=on`, the F32
branch of that fused path now goes through 1310's `bc_act_q81_try<op, true>` with the MUL node as output: the same
`op(x) * g` result plus MMVQ-padded native Q8_1 published for its MMVQ consumer, in one launch. Targets the remaining
Flash-Next misses fed by a gate multiply: the full-attention output gate (`attn_gated` -> wo) and the GatedDeltaNet
gated output norm (`final_output` -> ssm_out). Bit-identical F32; Q8_1 matches `quantize_q8_1`. Requires 1310.
Activation evidence: `BIGCHERRY_Q81 publish-act node=...(attn_gated-N)` under `BIGCHERRY_Q81_TRACE`.
GDN `final_output` (per-head rows of 128 read through `reshape_3d(head_v_dim*heads, T)`) is published flattened to the
consumer's padded row (1310 `flatten01`), matched by 1307's flattened-reshape lookup.
