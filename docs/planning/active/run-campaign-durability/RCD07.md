---
id: RCD07
order: 7
plan: run-campaign-durability
state: pending
created-at: '2026-09-26T00:52:12.716988+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P2
work: M
---

# Stage split and measurement-isolation qualification (scheduler-isolation-v1)

## Description

Split jobs into build / untimed correctness / timed measurement / ladder / production-lane / harvest stages. Builds may overlap untimed stages immediately; build-overlapping-timed-measurement only after a pre-declared A/A experiment passes (same binary both arms, idle vs one representative HIP compile, >=32 paired blocks per condition, randomized, production affinity/nice/ionice; accept iff condition effect 95% CI within +-0.10%, variance-ratio upper bound <= 1.15, no clock/power/thermal shift). Projected batch wall time 23.3 h -> ~14.6 h if it passes.

## Steps

1. Stage graph + resource claims (measure license/lock, build slots, GPU claims).
2. Run scheduler-isolation-v1 per architecture; record as non-patch evidence.
3. Enable overlap only on PASS; bound fixed before results.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Isolation evidence record; hardware batch wall-time comparison.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

## Change Log

- 2026-09-26T00:52:12.716988+00:00 (created-by): Created by agent
