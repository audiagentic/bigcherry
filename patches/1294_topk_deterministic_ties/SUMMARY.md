# 1294_topk_deterministic_ties

**Status:** untested
**Plan item:** RNX02

## What it does

The HIP parallel radix TOP_K picks tied columns at the cut by lowest column index (one extra block per row,
ballot/popcount prefix count) instead of atomic arrival order. `BIGCHERRY_TOPK_DETERMINISTIC=0` restores the
old path.

## Why

Qwen4Exp's QSA indexer scores are sums of ReLU, so many pooled KV blocks score exactly 0.0; past ~10K
context the top-k cut falls inside that tie and the attended blocks varied between runs. Greedy text
diverged across server starts at 32K and 80K with RCCL, cpu-root and host AllReduce alike.

Validation: tools/lab/flash-next/determinism.sh (cross-start greedy identity at 32K/80K, no MTP), decode and
prefill timing A/B, test-backend-ops TOP_K.

## Result (2026-10-03, flashnext-topk-ab-1, on 1291+1292)

Cross-start greedy identity, no MTP, cpu-root: 32K 317/317 chars identical (s1 vs s2), 80K 262/262 identical;
before 1294 the same probe diverged at character 45 (32K) and 0-1 (80K). Speed neutral: ~50 ms/MTP step at
10K and ~66 ms at 80K in all ABBA arms; with MTP at 10K three arms reached identical acceptance (351/478).
