# 1000_rdna4_mmq_q2k_q6k_fix: Upstream backport: RDNA4 MMQ codegen fixes for Q2_K and Q6_K

**Status:** validated
**Group:** upstream-fixes
**Plan item:** none

## What it does

Cherry-picks two narrowly-scoped fixes from unmerged upstream PR #25940 into the MFMA/WMMA MMQ vec_dot for Q2_K (forces a plain loop via #pragma unroll 1 to avoid a ROCm over-unroll/spill) and Q6_K (adds an explicit float cast before a scale multiply to change ROCm's codegen).

## Why

The PR's own numbers show large RDNA4 gains (Q6_K 1.90x, Q2_K 28.2x at n=512), and both quant types are in this project's own test corpus and hardware, so the fix is worth taking ahead of upstream merge rather than waiting.

## Upstream / provenance

Cherry-picked from open upstream PR https://github.com/ggml-org/llama.cpp/pull/25940. Deliberately excludes the PR's second change (a hand-written RDNA4 native-select heuristic), since this project's own tuner already measures candidates head-to-head per shape.
