# 1221_rd50_gdn_chunked_recurrence: Chunked fused GatedDeltaNet recurrence for RDNA3.5 (RD50, subsumes RD51/RD52/RD53)

**Status:** untested
**Group:** rdna-boosts
**Plan item:** RD50/RD51/RD52/RD53

## What it does

Folds the two new upstream-fork files (gated_delta_net_chunked.cu/.cuh) into gated_delta_net.cu, adding a GDN_CHUNK=32-token chunked kernel that turns the token-by-token delta-rule recurrence into dense matmuls over LDS on RDNA3.5, narrowly gated (scalar gate only, S_v==128, n_tokens>32, RDNA3_5) with an env-var force-disable for A/B benching.

## Why

AMD reports 1.89x GDN-op speedup and up to +20% E2E at ubatch>=4096 on the targeted shape. RD51 (DPP reduction)/RD52 (native exp2)/RD53 (launch-bounds tuning) are inline micro-decisions inside the same kernel body in the source PR, not independently portable hunks, so they are subsumed into this one patch rather than artificially split.

## Upstream / provenance

Ported from AMD-Ecosystem/llama.cpp PR #54 (merge commit 2b9497ff2, https://github.com/AMD-Ecosystem/llama.cpp). Fork-only work, no upstream mainline counterpart found.
