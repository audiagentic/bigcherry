---
id: QFP06
order: 0
plan: patching-qwen-flash-next
state: superseded
created-at: '2026-10-03T15:20:45.005526+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# Graph-instantiate OOM eviction + graph-cache-cap experiment (cap rejected; evidence retained)

## Description

QFP06 is retained as **negative graph-working-set evidence**, not an active graph-cap optimization.

Patch `1302_cuda_graph_oom_evict` remains a verified framework safety fix: on `cudaGraphInstantiate`/`hipGraphInstantiate` OOM, the backend destroys other cached `ggml_cuda_graph` instances, synchronizes, and retries once. It fixed the observed deep-fill production failure.

The proposed standing LRU cap was then tested by patch `1304_cuda_graph_lru_cap` and disproven as a performance strategy. The large cache is primarily the live decode working set created by tensor-split graph segmentation, not stale prefill debris. A cap materially below that working set forces eviction/recapture churn and regresses decode.

Therefore there is no remaining QFP06 implementation task. Any work to reduce the number of graph splits/instances belongs to the scheduler/AllReduce owners (QFP11/RNX11), while rank-arrival skew/load balance remains QFP09/QFP07. QFP13 may cite QFP06's memory result but must not create another graph-cache-cap plan.

## Evidence

- Deep fill near ~164K tokens hit graph-instantiation OOM before 1302; the OOM-evict/retry path recovered by evicting ~193 cached graphs on the constrained device.
- 1304 memory logging measured roughly ~2 MiB per cached graph instance.
- An ~80K-deep production request instantiated ~206 graphs per tensor-split GPU and reached roughly ~195 simultaneously cached instances (about 412/386/194 MiB across XTX0/XTX1/R9700 in the recorded run).
- The working set persisted in decode because `-sm tensor` splits at frequent AllReduce boundaries.
- Cap 32 caused continuous eviction/recapture churn and degraded the measured step time to ~65.4 ms/step versus ~48.8/46.7 ms/step uncapped (~37% regression in that experiment).

Conclusion: the ~400 MiB/GPU graph footprint is real working-set cost under the current split structure. Reclaiming it with a sub-working-set LRU trades memory for a large decode regression.

## Disposition

- Keep `1302_cuda_graph_oom_evict` as the verified OOM-recovery/safety mechanism.
- Keep 1304 graph-memory logging as diagnostic instrumentation where useful.
- Do **not** promote a standing graph cap below the live working set.
- Do not create another cache-cap/eviction plan from QFP13.
- If graph memory must be reduced, reduce the number of required live graph segments rather than evicting segments that will immediately be reused.

Canonical remaining owners:
- **QFP11 / RNX11:** graph segmentation / AllReduce capture-replay mechanics;
- **QFP09 / QFP07:** rank-arrival skew and per-class placement/load balance;
- **QFP13:** launch-count prioritisation only; QFP06 is supporting negative evidence.

## Validation / Reopen Gate

QFP06 should be reopened only if a future source/trace change materially changes the live graph working set (for example, graph segmentation is reduced enough that a bounded cap can exceed the steady-state working set while still reclaiming stale instances). Any renewed cap experiment must first measure the new working-set cardinality and choose a cap above it; do not repeat a below-working-set cap test as an optimization.

## Notes

Evidence runs: `flashnext-1302-deepfill`, `flashnext-v2-profile`, `flashnext-v2-1304-memlog`.

Related but not duplicate: PRBE66/PRBE67 cover graph keying/recapture correctness. QFP11 owns the remaining fewer-splits direction and has separately shown that AllReduce split/round-trip overhead is smaller than rank-arrival skew, so even graph consolidation is lower priority than load balance unless newer traces change that result.

## Change Log

- 2026-10-03T15:20:45.005526+00:00 (created-by): Created by agent
- 2026-10-03T16:19:51.201879+00:00 (updated-by): Updated: section:notes
- 2026-10-04 (agent): Superseded the active LRU-cap follow-up after 1304 proved cap-below-working-set recapture churn; retained 1302 OOM recovery and graph-memory measurements as evidence; assigned remaining split/skew work to QFP11/RNX11 and QFP09/QFP07.
- 2026-10-06T14:55:26.695084+00:00 (state-transition): State: superseded → superseded
