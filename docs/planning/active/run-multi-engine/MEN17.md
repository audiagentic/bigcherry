---
id: MEN17
order: 0
plan: run-multi-engine
state: pending
created-at: '2026-10-17T09:09:32.756838+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P1
work: M
---

# Post-bump hardware acceptance against stored baselines

## Description

Owner-approved (2026-10-17): after a bump, run each engine's declared acceptance configs (radiance: helper-ab speed, helper-kld and 27b-native-kld quality; llama.cpp: its production A/B) and compare against stored baselines with tolerances from measured noise, instead of reading numbers by eye. Includes named, stored KLD references per model/layout (no self-recorded reference in the same run) so the gate does not compare a control against its own reference.

## Steps

1. Baseline store: docs/reference/engines/<engine>/baselines/*.json (metric, value, noise band, run id, pin). 2. Engine declares acceptance configs + baselines in engine.toml. 3. `bigcherry --engine <name> accept` runs them through ab-benchmark and gates against baselines; writes an evidence record. 4. Stored KLD references (path + recorded-on pin/layout) used by quality configs. 5. pin-bump/pin-release can require a passing accept record.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

radiance accept on the 1.7.0 pin passes against the 1.3.0 baselines (2,050 pp4096 / 124 tg512 / KLD 0.064); a degraded arm fails.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

## Change Log

- 2026-10-17T09:09:32.756838+00:00 (created-by): Created by agent
