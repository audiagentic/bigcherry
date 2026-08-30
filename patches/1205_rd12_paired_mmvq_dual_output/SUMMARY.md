# 1205_rd12_paired_mmvq_dual_output: Fuse paired mmvq matmuls over a shared activation (RD12, dual-output)

**Status:** untested
**Group:** rdna-boosts
**Plan item:** RD12

## What it does

When two adjacent MUL_MAT nodes (e.g. attention K/V projections) share the same activation and output shape and are mmvq-eligible, computes both in one launch: the first matmul also computes the second weight's result as a fusion 'gate' and writes it to a separate dst_gate destination.

## Why

K and V projections in an attention layer run two mmvq matmuls over the identical activation; fusing them into one launch avoids re-reading the activation and issuing a second kernel. The fork reports bit-identical output vs the unfused path plus small positive tg64 gains on gfx1201.

## Upstream / provenance

Ported from stew675-rdna-boosts fork commit 44b51c66a (https://github.com/stew675/llama.cpp), with an adaptation to the base common.cuh struct shape (no RD17 field present). Not merged into ggml-org/llama.cpp master. Known incompatible with 1207 (RD17) if both are selected together.
