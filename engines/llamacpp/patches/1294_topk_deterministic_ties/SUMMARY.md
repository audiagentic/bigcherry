# 1294_topk_deterministic_ties

**Status:** validated
**Plan item:** RNX02

## What it does

The HIP parallel radix TOP_K picks tied columns at the cut by lowest column index (one extra block per row,
ballot/popcount prefix count) instead of atomic arrival order. `BIGCHERRY_TOPK_DETERMINISTIC=0` restores the
old path.

## PA44 packaging / identity proof

The two kernels and two file-local switches now live in `ggml/src/ggml-cuda/bc-topk-ties.cuh`. The header is
included at the exact old block position, immediately before `top_k_radix_cuda`, rather than at the top of
`top-k.cu`: these definitions require the file-local upstream `top_k_radix_state` and
`top_k_float_to_ordered` definitions above that point.

Identity audit: `_TIE_KERNEL` is moved byte-for-byte: 5,589 bytes, 151 physical lines, 142 non-blank lines.
Comparing the multiset of non-blank, whitespace-stripped lines across the touched `top-k.cu` plus the created file
gives 142/142 moved lines identical; removed-only lines = 0, added-only lines = 1, the single
`#include "bc-topk-ties.cuh"` line. Runtime launches, flags, evidence, and ENV_DOCS are unchanged.

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

## 2026-10-14 ordered-path dependency audit (QFP03 / BCOP202)

At pinned b11552, the default-on `top_k_radix_gather_ordered` runs after upstream counter-reset and atomic gather, but reads only the radix prefix/rank and original input; it rewrites every output slot without reading the counters or the previous gather output. The reset and atomic gather are therefore redundant **only when deterministic+ordered are both enabled**. QFP03 specifies one in-place guarded 1294 simplification; no code change or GPU speedup has been demonstrated. The tie-only and stock paths must retain the reset/gather. Host model: 6 fixed + 20,000 random comparisons passed; 11/11 source checks passed. Historical 2026-10-03 speed-neutral measurements predate ordered output (added 2026-10-07).

**Precision:** ordered output is deterministic **partitioned index order**, not globally ascending indices: all greater-than-cut indices ascending, then the lowest-index cutoff-equal indices ascending. Example `[0,2,0,3], k=3` -> `[1,3,0]`. This preserves consumer reduction order. No code/kernel changes in this documentation audit.
