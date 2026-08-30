# 1206_rd13_mul_mat_add_view_fusion: Fuse mul_mat + add through a view (reshape) node (RD13)

**Status:** untested
**Group:** rdna-boosts
**Plan item:** RD13

## What it does

Extends the existing mul_mat+add fusion in ggml_cuda_try_fuse to accept one RESHAPE node between the matmul and the add (using ggml_can_fuse_subgraph, verifying the view's src[0] is the matmul), instead of only matching an add directly after the matmul.

## Why

SSM models (e.g. qwen35moe) insert a reshape view between the output projection and the residual add, so the existing fusion never fired for them and every layer ran a separate add kernel.

## Upstream / provenance

Ported from stew675-rdna-boosts fork commit 0153d580d (https://github.com/stew675/llama.cpp). Not merged into ggml-org/llama.cpp master.
