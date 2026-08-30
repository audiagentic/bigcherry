# 1207_rd17_moe_topk_down_fold: Fold the MoE topk-weights MUL into the down projection (RD17)

**Status:** untested
**Group:** rdna-boosts
**Plan item:** RD17

## What it does

Adds an x_scale_channel_dst fusion flag so the mmvq kernel scales each result row by the destination channel's (token's) topk softmax weight directly in the matmul epilogue, instead of a separate broadcast MUL kernel run after the down projection.

## Why

The fork reports this removes 40 kernels per decode token on qwen35moe with bit-identical perplexity, by eliminating a separate elementwise-scale launch after every MoE down projection.

## Upstream / provenance

Ported from stew675-rdna-boosts fork commit 5e545b7da (https://github.com/stew675/llama.cpp). Not merged into ggml-org/llama.cpp master. Known anchor incompatibility with 1205 (RD12) if both are selected together; no production recipe combines them today.
