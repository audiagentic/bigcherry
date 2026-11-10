---
id: RHP21
order: 0
plan: patching-radiance-heat-placement
state: pending
created-at: '2026-11-10T13:12:05.775219+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P2
work: M
---

# First cold prompt after a documents-only period: shorten the live policy's adaptation lag

## Description

With the policy warmed on documents only, the first cold 27k prompt costs the static time (10.95 s) and the policy is at steady state (7.3-7.5 s) by the 3rd-4th prompt (~30-40 s of traffic, 4-5 placement cycles; runs/policy-lag, derived settings cadence 6.05 s, min_samples 2, min_residency 68.5 s). Candidates: faster heat evidence (shorter derived cadence where the measured transaction time allows), bounded early moves, predicted demand from the prompt. RHP17's speed-aware solver does not address this. Gate: no loss of steady documentation throughput, effect gate and quality unchanged.

## Steps

1. Replay offline (r3_policy_replay) over recorded heat series to see how many cycles a cold shift needs and which derived setting bounds it.
2. Evaluate options against the effect gate and residency rules; keep every setting derived from measurement.
3. If an option wins offline, A/B it on hardware with the document-then-cold protocol of runs/policy-lag.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Offline replay plus the policy-lag protocol: TTFT of the first cold prompts after documents, steady-state TTFT, document prefill, decode.

## Effort & Risk



## Standards



## Acceptance Criteria

Either the first-cold penalty shrinks measurably without regressing steady throughput or quality, or the lag is shown to be bounded by the derived transaction time and the item is closed.

## Notes

Source: gateway prior-work audit 2026-11-10 (missing item 3).

## Change Log

- 2026-11-10T13:12:05.775219+00:00 (created-by): Created by agent
