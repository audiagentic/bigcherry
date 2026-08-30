# 1203_rd050607_rdna4_wmma_fa_q6k_mmq: RDNA4 WMMA flash-attn and Q6_K mmq prefill performance work (RD05/RD06/RD07)

**Status:** untested
**Group:** rdna-boosts
**Plan item:** RD05/RD06/RD07

## What it does

Fixes a head-256 WMMA flash-attn combine race and a tile_Q reuse race, tunes and enables the WMMA flash-attn path up to head 576 on RDNA4 by default, and hoists/folds Q6_K mmq sub-scales into the row base-scale to remove an int-mul from the scale chain (fork: Q6_K mmq 40 -> 58 TFLOPS). Also adds op-timing instrumentation and test cases.

## Why

Targets specific RDNA4 kernel-level performance bugs and tuning gaps identified upstream in the fork's own commit series; the head-256 combine race and tile_Q reuse race are correctness fixes bundled with the performance tuning.

## Upstream / provenance

Ported (with adaptations for this project's own 1000/HI70 anchors) from stew675-rdna-boosts fork commit 1d525bd45 (https://github.com/stew675/llama.cpp), covering RD05/RD06/RD07. Not merged into ggml-org/llama.cpp master.
