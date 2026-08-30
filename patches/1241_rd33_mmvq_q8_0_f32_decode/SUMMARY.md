# 1241_rd33_mmvq_q8_0_f32_decode: Dense Q8_0 decode without activation quantization (RD33, AMD-MMV-001)

**Status:** untested
**Group:** rdna-boosts
**Plan item:** RD33

## What it does

Adds a defaulted f32_act template parameter to mul_mat_vec_q and a new vec_dot_q8_0_f32 device helper that dequantizes the Q8_0 weight block directly and dot-products it against the original F32 activation (F32 accumulation), skipping the Q8_1 activation-quantization stage entirely; gated to dense (non-MoE), Q8_0, ncols_dst==1, gfx1100, and only when nothing has been forced.

## Why

ggml_cuda_mul_mat_vec_q unconditionally quantizes the F32 activation to block_q8_1 before every MMVQ call, including plain single-token decode where there is no batching to amortize that extra kernel launch and pool allocation against, and the weight is already the only operand whose quantization matters for a bandwidth-bound matvec.

## Upstream / provenance

Local design, part of this project's own rdna-boosts experiment work (RD33), designed and verified against the real pinned source via dev-gpt-agent review. Not yet built, compiled, or measured on real hardware.
