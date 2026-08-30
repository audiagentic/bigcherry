# 1217_rd44_graph_opt_default_rdna35: Default GGML_CUDA_GRAPH_OPT to enabled on RDNA3.5 (RD44)

**Status:** untested
**Group:** rdna-boosts
**Plan item:** RD44

## What it does

Defaults the CUDA/HIP graph-optimization pass to enabled on gfx1151 (GGML_CUDA_CC_IS_RDNA3_5) while every other architecture keeps the previous env-var-gated-off default; GGML_CUDA_GRAPH_OPT stays an explicit override in both directions.

## Why

Last item in the AMD-STREAM chain (RD39->RD44): the fork measured +7.3% tg128 and +8.7% VLM decode on gfx1151 with graph-opt enabled, with no regression on small models.

## Upstream / provenance

Ported from AMD-Ecosystem/llama.cpp PR #56 (merge commit 6e9f948a0, https://github.com/AMD-Ecosystem/llama.cpp), with a bug fix (found in review) so the architecture check isn't wrongly cached process-wide across multiple GPU architectures. Depends on patches 1215/1216 (RD39-RD43) for correctness once triggered.
