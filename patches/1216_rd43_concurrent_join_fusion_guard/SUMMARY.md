# 1216_rd43_concurrent_join_fusion_guard: Keep a concurrent-region join node out of op-fusion (RD43)

**Status:** untested
**Group:** rdna-boosts
**Plan item:** RD43

## What it does

Temporarily lowers cgraph->n_nodes around ggml_cuda_try_fuse while a concurrent region is active, so op-fusion cannot absorb the join add node that rejoins an auxiliary stream; a GGML_ASSERT catches any future fusion pattern that still crosses the join.

## Why

Without this guard, op-fusion could absorb RD42's (patch 1215's) shared-expert join add as a preceding matmul's bias-add, so the join handler never runs, the aux stream is never rejoined, and CUDA/HIP graph capture aborts with 'capturing stream has unjoined work'.

## Upstream / provenance

Ported from AMD-Ecosystem/llama.cpp PR #71 (merge commit 0f0db6292, https://github.com/AMD-Ecosystem/llama.cpp). Fork-only work; apply after patch 1215.
