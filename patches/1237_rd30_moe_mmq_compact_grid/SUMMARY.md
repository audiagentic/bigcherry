# 1237_rd30_moe_mmq_compact_grid: Compact the MoE MMQ launch grid (RD30, AMD-MOE-001)

**Status:** untested
**Group:** rdna-boosts
**Plan item:** RD30

## What it does

Adds a prep kernel (mmq_build_moe_block_map) that flattens (expert, expert-local-tile) pairs into one linear grid dimension sized to the real total tile count, instead of upstream's rectangular grid sized from the worst-case expert width times n_expert; falls back to the exact legacy grid whenever the compact map would exceed grid or shared-memory limits. Gated to gfx1100 exactly.

## Why

Upstream's non-stream-K MMQ launch gives every MoE expert the same worst-case tile-column count regardless of its real routed-token occupancy, launching far more blocks than needed on real production models (confirmed via rocprofv3 on Qwen3.6-35B-A3B, n_expert=256); AMD's own PR #63 reports a modest +1.9-5.4% prefill gain from compacting this.

## Upstream / provenance

Concept ported from AMD-Ecosystem/llama.cpp PR #63 (fork-only, https://github.com/AMD-Ecosystem/llama.cpp), redesigned against the real b10502 source rather than ported verbatim. Extensive informal real-hardware evidence gathered, but STATE stays untested pending the project's own formal HI83-governed validation campaign.
