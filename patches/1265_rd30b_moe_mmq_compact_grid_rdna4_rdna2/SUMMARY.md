# 1265_rd30b_moe_mmq_compact_grid_rdna4_rdna2

**Status:** untested
**Plan item:** RD30

## What it does

Widens 1237's compact MoE MMQ launch-grid gate from gfx1100 only to also admit RDNA4 (gfx1201) and RDNA2 (gfx1030).

## Why

1237 measured +7.3..7.4% MoE prefill on gfx1100 with byte-identical output. The compaction is host-side launch geometry, so the same empty per-expert launches exist on the other cards; this package measures it there without touching 1237's own evidence.

## Upstream

Extension of 1237 (AMD-Ecosystem/llama.cpp PR #63 concept). Requires 1237.
