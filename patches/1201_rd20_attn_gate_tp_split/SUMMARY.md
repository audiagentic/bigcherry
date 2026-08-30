# 1201_rd20_attn_gate_tp_split: Align attn_gate tensor-parallel split granularity with attn_q (RD20)

**Status:** superseded
**Group:** rdna-boosts
**Plan item:** RD20

## What it does

Gave attn_gate.weight the same head-aligned split granularity as attn_q in the regular-attention branch of get_split_granularity, matching what the recursive-attention (pattern_qkv) path already did.

## Why

Superseded: this correctness fix (with 3+ GPUs, a mismatched attn_gate split granularity could round a device's share to zero and abort the element-wise MUL between attention output and gate) is replaced by upstream PR #27574, which lands the equivalent fix in mainline llama.cpp.

## Upstream / provenance

Originally ported verbatim from stew675-rdna-boosts fork commit 3b200b259 (https://github.com/stew675/llama.cpp). Superseded by mainline upstream PR #27574.
