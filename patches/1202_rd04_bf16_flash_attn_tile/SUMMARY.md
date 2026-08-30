# 1202_rd04_bf16_flash_attn_tile: Native-BF16 flash-attn tile kernel series (RD04, net of 7 fork commits)

**Status:** untested
**Group:** rdna-boosts
**Plan item:** RD04

## What it does

Makes the flash-attn TILE kernel read BF16 K/V natively on RDNA3+ with FP32 accumulation (rather than F16 accumulators that flush small values at deep context), adds a packed BF16 PV path, and stores the softmax KQ buffer as BF16 to halve SRAM use; non-RDNA3 targets keep the existing F16 tile path.

## Why

The fork reports better perplexity than F16 KV cache at deep context (PPL 15.18 BF16 vs 15.59 F16 vs 14.97 F32 @ 32k) by avoiding F16 accumulator flush on small values.

## Upstream / provenance

Generated as the net diff of seven contiguous stew675-rdna-boosts fork commits (8623179e1..0581c532c, https://github.com/stew675/llama.cpp), since intermediate states cancel out. Not merged into ggml-org/llama.cpp master.
