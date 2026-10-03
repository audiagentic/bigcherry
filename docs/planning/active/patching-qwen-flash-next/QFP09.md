---
id: QFP09
order: 0
plan: patching-qwen-flash-next
state: pending
created-at: '2026-10-03T15:21:02.481672+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P2
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

## Change Log

- 2026-10-03T15:21:02.481672+00:00 (created-by): Created by agent
