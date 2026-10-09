# 1241_rd33_mmvq_q8_0_f32_decode: Dense Q8_0 decode without activation quantization (RD33, AMD-MMV-001)

**Status:** validated
**Plan item:** RD33

## What it does

Adds a defaulted f32_act template parameter to mul_mat_vec_q and a new vec_dot_q8_0_f32 device helper that dequantizes the Q8_0 weight block directly and dot-products it against the original F32 activation (F32 accumulation), skipping the Q8_1 activation-quantization stage entirely; gated to dense (non-MoE), Q8_0, ncols_dst==1, gfx1100, and only when nothing has been forced.

## Why

ggml_cuda_mul_mat_vec_q unconditionally quantizes the F32 activation to block_q8_1 before every MMVQ call, including plain single-token decode where there is no batching to amortize that extra kernel launch and pool allocation against, and the weight is already the only operand whose quantization matters for a bandwidth-bound matvec.

## Upstream / provenance

Local design, part of this project's own rdna-boosts experiment work (RD33), designed and verified against the real pinned source via dev-gpt-agent review.

## Lifecycle note

`state = "validated"` (2026-09-30), scoped to gfx1100 (dual 7900 XTX, `-sm tensor`), dense Q8_0, `ncols_dst == 1` decode. Contract `RD33-MMVQ-Q8_0-F32-DECODE` status=pass over 4 sessions (`tierL-qwen27b-q8` positive tg128 +4.56..+4.67%, 10/10 pairs each; `tierA-qwen4b-q6k` non-firing control within the no-regression bound; CPU-reference `test-backend-ops` MUL_MAT q8_0 50/50; trace-marker activation at ncols=1 only). Final sign-off by dev-gpt-agent. See README.md "Promotion evidence".

Not claimed: other architectures or models, MTP verify batches (ncols>1 keep the stock path; acceptance parity 0.90101 vs control), prefill. The Q6_K control moved +1.4% in every session (build-layout effect, not attributable to this patch); the control-adjusted decode effect is about +3.2%. Not yet added to a recipe patch-set -- separate decision.
