---
id: QFP06
order: 0
plan: patching-qwen-flash-next
state: pending
created-at: '2026-10-03T15:20:45.005526+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# 1302 graph-instantiate OOM eviction + capped graph cache (VRAM reclaim)

## Description

Patch 1302_cuda_graph_oom_evict (framework fix, verified): on cudaGraphInstantiate/hipGraphInstantiate OOM the backend context destroys every other cached ggml_cuda_graph, syncs and retries once. Fixes a production crash: at -c 196608 a ~164K-token fill OOMed in hipGraphInstantiate on the R9700 (production build too). Follow-up: a standing LRU cap on cached graph instances per context (193 were live on the R9700) to return that VRAM to KV/compute permanently.

## Steps

1. Measure per-instance VRAM (hipMemGetInfo deltas around instantiate). 2. Implement LRU cap (env, e.g. 32). 3. Check decode cost of recapture churn. 4. Re-run max-context search with the cap.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Deep fill near max ctx without eviction storms; decode ms/step unchanged; freed VRAM converted into context or ub1024.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Evidence: flashnext-1302-deepfill (164,470 tokens, 636 pp / 46.4 tg, evicted=193 on device 2); profile v2 deep fill also evicted 193 at 164K. Upstream only evicts graphs unused for 10 s. Related but distinct: PRBE66/PRBE67 (graph keying/recapture correctness). GPT review incl. LRU code + per-instance memory measurement: req_ee409a9b21e74e86.

2026-10-04 patch 1304_cuda_graph_lru_cap (BIGCHERRY_CUDA_GRAPH_CAP, BIGCHERRY_GRAPH_MEMLOG) measured on profile v2 (flashnext-v2-1304-memlog, -cap32-d24k): each cached graph instance holds ~2 MiB; an 80K-deep request instantiates 206 graphs per tensor-split GPU, max 195 cached (412/386/194 MiB on XTX0/XTX1/R9700, draft GPU 3 graphs). The working set is DECODE, not leftover prefill: under -sm tensor each step's graph is split at every AllReduce (~96 per step), each split cached separately. Cap 32 = eviction/recapture churn: 65.4 ms/step vs 48.8/46.7 uncapped (+37%). Conclusion: an LRU cap below the working set is a loss; the ~400 MB/GPU is real working set. The real lever is fewer, larger graphs (AllReduce captured inside a graph instead of splitting it), which would also attack the ~56-61% GPU idle gaps (QFP09). Memlog stays useful as an instrument.

## Change Log

- 2026-10-03T15:20:45.005526+00:00 (created-by): Created by agent
- 2026-10-03T16:19:51.201879+00:00 (updated-by): Updated: section:notes
