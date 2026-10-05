---
id: BCOP08
order: 8
plan: bc-optimizations
state: superseded
created-at: '2026-10-05T04:37:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: L
---

# Assess miss-cost-aware multi-tier expert residency

## Description

Follow-through for MET01/MET05 audit. MET01 should own residency decisions while MET05/1328 owns auxiliary-device execution. Evaluate static primary residency, persistent 6900-XT warm tier, upstream-style GPU LRU for host misses and CPU fallback by measured critical-path milliseconds saved per resident byte, not hit rate alone.

## Steps

1. Finish/qualify 1328 whole-layer ROCm3 execution before expert-granular 6900 placement.
2. Qualify llama.cpp #29887 cache callbacks on the actual BigCherry scheduler/meta configuration and prove callbacks/entries are active; a flag with zero active cache entries is FAIL.
3. Measure equal-VRAM static-residency vs cache arms including PP, TG, hit/miss, migration bytes/time and displaced resident layers.
4. Build `(layer,expert)` service-time/GiB scores from measured route probability, transfer, compute, queue and migration costs.
5. Add the ROCm3 persistent tier only if whole-layer hardware results improve end-to-end service time per GiB.
6. Keep cache policy in MET01 and execution/transport semantics in MET05/1328.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation



## Effort & Risk



## Standards



## Acceptance Criteria

- #29887-style cache activation is proven or rejected on production graph configuration.
- Equal-VRAM comparisons include both PP displacement cost and TG benefit.
- 6900 tier is admitted only from positive whole-layer hardware evidence.
- No second cache/router/placement registry is introduced.

## Notes

2026-10-05 review: superseded by owner item MET01 (carries #29887 cache qualification, LRU and per-GiB scoring) with execution in MET05/1328. Not implemented; the 1328 hardware sweep has not been run. #29887 is not in pin 050439614.

## Related

MET01, MET04, MET05, patch 1328; llama.cpp #29887.

## Change Log

- 2026-10-05T04:37:06.226829+00:00 (updated-by): Updated: section:notes
- 2026-10-05T04:37:51.734654+00:00 (state-transition): State: pending → superseded
