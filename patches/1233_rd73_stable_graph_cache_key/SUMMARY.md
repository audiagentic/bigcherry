# 1233_rd73_stable_graph_cache_key: Replace the HIP/CUDA graph-cache key with a stable FNV-1a shape fingerprint (RD73, re-scoped from FORK-MTP-003)

**Status:** untested
**Group:** rdna-boosts
**Plan item:** RD73

## What it does

Replaces ggml_cuda_graph_get_key()'s use of the raw first-node pointer as the cuda_graphs map key with a 64-bit FNV-1a fingerprint over node count plus the first/last nodes' op/name/ne[], which is stable across allocations for a recurring shape; the existing per-node memcmp correctness check in ggml_cuda_graph_update_required() is unchanged, so a fingerprint collision only costs an extra recapture.

## Why

The raw first-node pointer is allocation-dependent, so a fresh allocation for an otherwise-identical recurring shape (e.g. repeated speculative-verify batches) caused a cold cache miss almost every time even though the shape hadn't changed; the fork measured a verify ubatch sync drop from 150ms to 57ms on a 3.8k-node graph.

## Upstream / provenance

Ported byte-for-byte from mrlordcat-rdna-lab commit 7f2e7e4a3 (https://github.com/MrLordCat/llama.cpp-rdna-lab), after an external review caught and fixed a bug in an earlier draft (hashing the whole fixed name buffer instead of its used length). Not merged into ggml-org/llama.cpp master.
