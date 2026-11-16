# rad0029_rank_own_split

## Summary

Radiance sizes one VRAM split for every rank: the smallest card's capacity, the largest rank's weights and the dearest
rank's cache. One block table serves every rank, so the common token count is the least any rank pages. Ranks whose
cache is cheaper per token are re-split under `expert_tiered`. Nothing re-splits the rank that limits the token count,
even when it has memory to spare.

At TP3 over two RX 7900 XTX (24 GB) and an R9700 (32 GB), the R9700 holds 2 of the 4 KV heads and pages the fewest
tokens. Its card is 8 GB larger, and with a split that gives it fewer weights, its weights pool also has room left over.
None of that reached the cache: after a load, about 8 GiB of the R9700 sat free.

This patch adds a second re-split, which runs when every rank's model is dense, under any `--placement`. Each rank gets:

- a weights pool of exactly its own weights;
- a KV pool of the common pool, plus what its weights pool gave back, plus the free memory it measured above the smallest
  card's.

The common token count becomes the least that any rank pages in its own pool. Each KV pool is then cut to the least whole
MiB that pages that count, so the XTX ranks keep the rest of their memory free. If that count is no larger than the
common split's, nothing changes.

It does not apply when:

- any rank has an expert plane;
- the device is unified memory;
- `--gpu-memory-mib` was given;
- a rank's weights do not all fit the common weights pool.

Each re-split rank logs one `kv: rank N's own split` line.

## Outcome

Untested (not promoted). VTP33, MXFP4 Qwen3.8-27B, 262,144-token context, 8 sequences:

- **6/6/12 split:** 917,184 → 1,435,072 tokens carved, which is 3.5 → 5.5 full-length sessions. TP2 holds 3.3.
  Throughput at 4/6/8 streams was unchanged.
- **7,7,2 split:** 993,536 → 1,743,616 tokens, which is 6.7 sessions.
- TP2 on two alike cards is unchanged. Greedy answers were identical.
- 8 concurrent 128k-token prompts (1.04M tokens) ran without errors.

In a dense world it replaces the cheaper-rank re-split (`expert_tiered`, which the launcher's manifests select). That
re-split can only shrink a cheaper rank's KV pool, since its weights pool cannot grow past its weights, so it frees
nothing. The first version of this patch skipped any world where that re-split had run, which left the launcher path
without the gain.

## Detail

Files:


- `core/engine_bringup.cpp`: `Engine::plan`, after the cheaper-rank re-split.

Tests: `tools/tests/patch/test_vtp33_rank_own_split.py`: metadata and recipe, edit content, guards, and applying the chain to
the pinned radiance v1.17.0 source.

### Promotion record

Model: Qwen3.8-27B MXFP4 and int8 (.rad), TP3 over two RX 7900 XTX and an R9700, DFlash2 drafter for rad0028
Mechanics: the patch's tests in tools/tests/patch pass and patch-lint is clean on the promotion PR; the lane builds carrying it (VTP logs named below) build and serve.
Activation: rad0028 all-gathers the drafted tokens and every rank proposes from rank 0's (the cross-rank disagreement diagnostic found 166 differing rows on rank 2 before; VTP31.md experiment 13); rad0029 reports the per-rank KV pool above the common pool (VTP33.md).
Identity: rad0029 greedy answers identical and TP2 on two alike cards unchanged (VTP33.md); rad0028 is a fault fix: int8 TP3 stress runs faulted 11 of 22 before, 0 of 12 with rad0028 (VTP31.md).
A/B: rad0028 2x ABBA 1/4/6/8 streams 83.4/189.6/179.9/213.4 before vs 83.8/188.5/179.8/212.8 tok/s after (within 0.6%), TTFT +0.2%; rad0029 holds 5.5 full-length 262k sessions against TP2's 3.3 (+67%) at unchanged 4/6/8-stream throughput (VTP33.md, logs cap-1812 and cap-1839).
