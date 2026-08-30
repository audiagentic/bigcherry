# 1215_rd394041_amd_stream_moe_overlap: Honor active HIP stream, per-stream cuBLAS handles, dedicated concurrent scratch, and MoE shared-expert overlap (RD39/RD40/RD41/RD42)

**Status:** untested
**Group:** rdna-boosts
**Plan item:** RD39/RD40/RD41/RD42

## What it does

Fixes ggml_cuda_op_mul_mat to honor the assigned stream instead of hardcoding stream 0 and gives every (device,stream) pair its own cuBLAS handle (RD39/RD40); replaces in-place branch-node interleaving with a dedicated, reused scratch buffer for concurrent graph regions (RD41); and uses that infrastructure to detect the MoE shared-expert join and run the shared expert on an auxiliary stream during small-batch decode (RD42).

## Why

The single per-device cuBLAS handle's workspace corrupted concurrent GEMMs sharing it, and the pre-existing branch-node interleaving for concurrency was fragile against allocator reuse. Ported as one net patch because RD40/RD41 are declared prerequisites of RD39, and RD42 of all three, in the plan items themselves; splitting them would leave the graph optimizer in an uncompilable intermediate state. The fork measured +7.4% tg128 on Qwen3.6-35B-A3B Q4_K_M.

## Upstream / provenance

Ported from AMD-Ecosystem/llama.cpp PR #36 (merge commit 367c4d04f, https://github.com/AMD-Ecosystem/llama.cpp). Fork-only work, not ancestral to mainline or this project's pin.
