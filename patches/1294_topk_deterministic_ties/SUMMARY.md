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
