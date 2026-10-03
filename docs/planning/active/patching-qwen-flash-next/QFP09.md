---
id: QFP09
order: 0
plan: patching-qwen-flash-next
state: pending
created-at: '2026-10-03T15:21:02.481672+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P0
work: M
---

# Per-rank critical-path census by op class (R9700 slow rank) -> selective per-class skew

## Description

The R9700 (gfx1201, PCIe x4) is the slow tensor-split rank: less R9700 share is consistently faster (2.3,2.3,2.4 / 2.4,2.4,2.2: -3% ms/step vs 2,2,3 at 128K), while AllReduce payload is split-independent, pointing at R9700 compute kernels rather than collectives. Time each rank's compute-finished -> collective-finished span by op class (MoE experts, shared expert, GDN, attention, elementwise); if one class dominates, skew only that class (as 1303 does for attention) instead of the global -ts.

## Steps

1. rocprofv3 kernel trace per GPU on profile v2 decode; bucket by op class. 2. Identify dominant R9700 class. 3. Per-class split override (generalise 1303's mechanism, e.g. BIGCHERRY_FFN_TS).

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

ms/step ABBA at matched context; no context loss.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Split screens 2026-10-03 (QFN01 notes): 2,2,3.4 49.3; 2,2,3.8 49.1; 1.9,2.1,3 48.1; 2.1,2.1,2.8 48.0; 2.2,2.2,2.6 47.4; 2.3,2.3,2.4 46.8; 2.4,2.4,2.2 46.8 ms/step vs ~48.3. Global skew loses context (2.3,2.3,2.4 fits only 160K q8). GPT RV4215 rank #4. Tools: tools/lab/flash-next/long-ctx-profile.sh perf/timing/synctrace modes, sync-tracer.c.

2026-10-04 profile v2 decode census (flashnext-v2-profile, rocprofv3, tools/lab/flash-next/rank-census.py; per generated token, profiler-slowed). GPUs are busy only 39-44% of wall at every depth: ~56-61% of each token has no kernel running (host scheduling/sync/launch gaps) - the largest single lever. Slow rank flips with depth because 1303 put all attention on the two XTX: ~10K: R9700 slowest (moe-mmvq 1.42 vs 0.85 ms/tok on XTX at 42% vs 27-31% expert share), XTX wait ~1 ms/tok more in allreduce. ~80K: XTX flash-attn 1.46-1.53 ms/tok, R9700 now waits (allreduce 3.17 vs 2.5-2.75). ~160K: XTX flash-attn 3.4-3.7 ms/tok, R9700 allreduce wait 5.7 vs 2.2-3.0 ms/tok -> ~3 ms/tok of R9700 idle. Kernels/token ~1300-1410 per GPU (elementwise ~370-400, quantize ~180-200, mmvq ~150-166). Draft (6900): 1.2 ms/tok at 10K, 3.1 at 160K (attention 1.5). Next: (a) synctrace/apitrace modes on profile v2 to attribute the idle gaps; (b) depth-aware attention placement (QFP07): give the R9700 attention share at long context (rotated heads), trading memory headroom; (c) expert share off the R9700 at short context where VRAM allows.

2026-10-04 promoted to P0 by the QFP11 boundary decomposition: per-AR arrival skew (median ~80 us, R9700 last in ~70% of ARs; p90 800 us at 80K when XTX attention is slow) costs ~2.4 ms/token, far more than split boundaries (~0.4). Cause: -ts 0.31,0.27,0.42 sizes the R9700's expert share by VRAM (42%) while its memory bandwidth is ~2/3 of an XTX. Next code: patch 1305 BIGCHERRY_FFN_TS - a second split vector for the MoE expert family (ffn_*_exps, and shexp if it splits) on the 1303 mechanism, so expert placement follows bandwidth within VRAM limits; screen at 10K/80K with the skew metric from ar-boundary.py as the activation/benefit evidence.

2026-10-04 1305 (BIGCHERRY_FFN_TS) screens on profile v2 placement at -c 65536 (-c 131072 OOMs XTX0 with more expert share): ~24K: 0.34,0.33,0.33 45.8 vs 46.0/45.6 ms/step (neutral); 0.37,0.35,0.28 47.1 vs 45.8/46.0 (+2.6%, worse); ~40K: 0.34,0.33,0.33 49.3 vs 50.1/49.5 (-0.8%, noise). Conclusion: moving the expert family alone off the R9700 does not remove the arrival skew - the XTX become the slow side once they take more experts, and the R9700 still carries 42% of the -ts tensors (GDN/recurrent projections, other matvec ~1.9 ms/token). The 'R9700 is slow because of expert share' hypothesis is not supported. Next: per-segment attribution - for each AR interval, which kernels on the last-arriving GPU account for its lateness (extend ar-boundary.py to sum kernel classes between consecutive ARs per GPU), then move only that class. 1305 parked (mechanism works and is cheap; no winning vector found).

2026-10-04 per-segment attribution (tools/lab/flash-next/ar-segment.py, ARs aligned across GPUs by arrival time - ordinal alignment was off by one and is unreliable; ar-boundary.py has the same weakness for its per-AR skew numbers). R9700 last at 76% (~10K) / 85% (~80K) of ARs. Extra time on the late GPU vs the first (ms/token): ~10K moe-mmvq 0.68, mmvq 0.52, mmvf 0.17, idle gaps inside the segment 0.66; ~80K moe-mmvq 1.00, mmvf 0.23, mmvq 0.13. Per share the R9700 runs expert matvec ~22% slower than an XTX (bandwidth). Expert-only balance point ~FFN_TS 0.355,0.355,0.29, but the XTX also carry all attention (1303) and become late - consistent with 1305's 0.37,0.35,0.28 being +2.6% and 0.34,0.33,0.33 flat. Conclusion: the current split is near the trade-off optimum; split balancing has hit diminishing returns (a mild shift might give 1-2%, needs a full ABBA to resolve). New lead: 0.66 ms/token of idle gaps inside the R9700's own segments at short context (per-kernel launch/dispatch gaps on the gfx1201 rank, PCIe x4) - check kernel-to-kernel gap distribution per GPU.

## Change Log

- 2026-10-03T15:21:02.481672+00:00 (created-by): Created by agent
- 2026-10-03T16:05:27.527362+00:00 (updated-by): Updated: section:notes
- 2026-10-03T16:35:24.607549+00:00 (updated-by): Updated: priority='P0', section:notes
- 2026-10-03T16:59:58.379545+00:00 (updated-by): Updated: section:notes
- 2026-10-03T17:25:54.754317+00:00 (updated-by): Updated: section:notes
