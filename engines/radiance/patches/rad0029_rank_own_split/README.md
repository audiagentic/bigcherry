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
