# 1234_rd58_pin_state_buffer_multigpu_restore: Pin the host state buffer during multi-GPU prompt-cache/checkpoint state restore (RD58, UP-HIP-003)

**Status:** untested
**Group:** rdna-boosts
**Plan item:** RD58

## What it does

Registers the state-restore buffer in llama_context::state_seq_set_data() as portable pinned host memory (RAII-scoped, host-only path, >=1 MiB threshold) around the async H2D copy, and flattens any Meta-typed scheduling wrapper into its real per-device backends before the registration loop so pinning actually activates under -sm tensor, not just -sm layer.

## Why

ROCm/rocm-systems#4817 is a real, still-open runtime defect where an async H2D copy from pageable host memory can fault the SDMA engine when 2+ HIP devices share a process; pinning the buffer keeps the mapping stable. Upstream closed its own PR redirecting to the ROCm issue rather than disputing the fix, so this application-layer workaround remains the only available mitigation.

## Upstream / provenance

Ported (with a failure-logging deviation and a real activation-gap fix for -sm tensor topologies) from upstream PR #27405 (closed, not merged, https://github.com/ggml-org/llama.cpp).
