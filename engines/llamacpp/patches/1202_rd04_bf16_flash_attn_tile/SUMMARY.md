# 1202_rd04_bf16_flash_attn_tile: Native-BF16 flash-attn tile kernel series (RD04, net of 7 fork commits)

**Status:** untested
**Plan item:** RD04

## What it does

Makes the flash-attn TILE kernel read BF16 K/V natively on RDNA3+ with FP32 accumulation (rather than F16 accumulators that flush small values at deep context), adds a packed BF16 PV path, and stores the softmax KQ buffer as BF16 to halve SRAM use; non-RDNA3 targets keep the existing F16 tile path.

## Why

The fork reports better perplexity than F16 KV cache at deep context (PPL 15.18 BF16 vs 15.59 F16 vs 14.97 F32 @ 32k) by avoiding F16 accumulator flush on small values.

## Upstream / provenance

Generated as the net diff of seven contiguous stew675-rdna-boosts fork commits (8623179e1..0581c532c, https://github.com/stew675/llama.cpp), since intermediate states cancel out. Not merged into ggml-org/llama.cpp master.


## 2026-10-11 qualification disposition

Still `untested`. The b11474 RDNA3/4 selector sends masked GQA=4/8 pp512 head 64/128/256 to MMA_F16, **not** the native BF16 TILE branch in this patch. BF16 KV flags and PPL parity alone cannot prove 1202 activation or full-vocabulary backend-reference correctness. The patch has no trace marker; the producer aliases both named correctness checks to one PPL result. Nine tracked records are ineligible, and b11126 PPL failures are missing parseable output, not proven numerical mismatches. Reuse the existing 1202 package and producer; gate new GPU A/B on source-matched selector fixture, executed BF16 TILE marker, true logits/KLD reference and exposed-cost measurement. See PRBE01 / BCOP132. No hardware or build qualification was performed in this audit.
