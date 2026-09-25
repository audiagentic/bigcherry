# 1207_rd17_moe_topk_down_fold: Fold the MoE topk-weights MUL into the down projection (RD17)

**Status:** rejected
**Plan item:** RD17

## What it does

Adds an x_scale_channel_dst fusion flag so the mmvq kernel scales each result row by the destination channel's (token's) topk softmax weight directly in the matmul epilogue, instead of a separate broadcast MUL kernel run after the down projection.

## Why

The fork reports this removes 40 kernels per decode token on qwen35moe with bit-identical perplexity, by eliminating a separate elementwise-scale launch after every MoE down projection.

## Upstream / provenance

Ported from stew675-rdna-boosts fork commit 5e545b7da (https://github.com/stew675/llama.cpp). Not merged into ggml-org/llama.cpp master. Known anchor incompatibility with 1205 (RD12) if both are selected together; no production recipe combines them today.

## Rejected 2026-09-26

Four sessions per architecture at pin b11126 (contract RD17-MOE-TOPK-DOWN-FOLD):
correctness failed every session (full-vocab backend reference diverged at
decode step 55, logprob diff 1.8 on a ~1e-13-probability token), and the target
lane regressed (tg128 -0.1..-0.2% gfx1100, -1.0% gfx1201). Owner-approved; GPT
review req_4d131e8b7c1c452d concurred (the correctness divergence alone
disqualifies it). Evidence retained in evidence/validation.json.
