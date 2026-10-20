# 1237_rd30_moe_mmq_compact_grid: Compact the MoE MMQ launch grid (RD30, AMD-MOE-001)

**Status:** validated
**Plan item:** RD30

## What it does

Adds a prep kernel (mmq_build_moe_block_map) that flattens (expert, expert-local-tile) pairs into one linear grid dimension sized to a host-computable upper bound of the real tile count (padded blocks exit on-device), instead of upstream's rectangular grid sized from the worst-case expert width times n_expert; falls back to the exact legacy grid whenever the compact map would exceed grid or shared-memory limits. Gated to gfx1100 exactly.

## Why

Upstream's non-stream-K MMQ launch gives every MoE expert the same worst-case tile-column count regardless of its real routed-token occupancy, launching far more blocks than needed on real production models (confirmed via rocprofv3 on Qwen3.6-35B-A3B, n_expert=256); AMD's own PR #63 reports a modest +1.9-5.4% prefill gain from compacting this.

## Upstream / provenance

Concept ported from AMD-Ecosystem/llama.cpp PR #63 (fork-only, https://github.com/AMD-Ecosystem/llama.cpp), redesigned against the real b10502 source rather than ported verbatim. Earlier informal hardware evidence preceded the formal b11126 four-session HI83 contract PASS; manifest and validated-enhancements now mark this package validated for gfx1100. The host grid still launches an upper bound, not exactly the device-computed real tile count.
