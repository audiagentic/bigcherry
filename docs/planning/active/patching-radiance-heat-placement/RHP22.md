---
id: RHP22
order: 0
plan: patching-radiance-heat-placement
state: pending
created-at: '2026-11-10T13:12:09.846895+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P2
work: S
---

# Live placement quality disposition: the forced-churn KL p99 breach (+0.234 against +0.2)

## Description

The forced-churn KL gate on the current build reproduces the 2026-11-08 corrected result to every digit: mean +0.008, p99 0.704 -> 0.938 (+0.234 over the +0.2 limit), p99.9 +0.43, top-1 -0.6 points; the other thresholds pass. RHP16 attributes the p99 breach to hot layer-0 experts on the 6900 XT tail (RAD46 tail accuracy) and the policy moves hot experts off the tail. Performance gains do not pass a quality gate: the accepted-quality disposition must be explicit. Note the gate scores one swapped placement (swaps finish at engine start-up), not ownership changes mid-scoring.

## Steps

1. Decide: waive with the stated mechanism and evidence, or add a gate that scores the policy's actual final placement (a KL run on the placement a warmed policy settles on) and mid-scoring moves.
2. If a real-placement KL run is feasible (the policy cannot move inside a 15-17 s scoring pass), record how the placement is frozen from a warmed run.
3. Record the disposition in RHP16 and the guide.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

KL numbers for the chosen scoring protocol beside the static control, with thresholds.

## Effort & Risk



## Standards



## Acceptance Criteria

A recorded accept/waive/fail decision for the policy's numerical quality, with evidence, that a later reader can find from LIVE_PLACEMENT.md.

## Notes

Source: gateway prior-work audit 2026-11-10 (missing item 4); runs/policy-kld.

## Change Log

- 2026-11-10T13:12:09.846895+00:00 (created-by): Created by agent
