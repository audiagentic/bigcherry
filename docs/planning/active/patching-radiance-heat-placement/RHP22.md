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

2026-11-11 cross-ISA evidence: flashnext-2xtx-rdna2-tail forced-churn KL (runs/rhp20-tail-kld, 1 pair, 8,240 rows, 132 swaps committed during scoring) passes every threshold: p99 +0.112 (limit +0.2), p99.9 +0.295, mean +0.0069, top-1 +0.07 pt. The flashnext-4gpu breach (+0.234) is therefore layout-specific, consistent with the RHP16 mechanism (hot layer-0 experts on the 6900 XT tail of the 4-GPU layout). Decision still open: waive on mechanism vs score a frozen warmed placement.

## Update 2026-11-11: the breach is deterministic, not run-to-run noise

Three-arm quality-only run on the current build (flashnext-4gpu, 8,240 rows, rad116 BF16 reference): `policy-off`, an identical `policy-off-repeat`, and `forced-churn`. The repeat arm is bit-identical to the control (mean 0.0660, median 0.0154, p99 0.7036, p99.9 3.0029, top-1 88.54, ppl 8.9944), so the measurement noise band on this harness is zero and the same-reference repeat asked for in Step 1 is answered. Forced churn: mean 0.0738 (+0.008), median 0.0171 (+0.0017), **p99 0.9377 (+0.234 against the +0.2 limit, fails)**, p99.9 3.4331 (+0.43, passes the +0.5 limit), top-1 87.94 (-0.6 points, passes), ppl 8.837. Only p99 breaches, reproducibly, so "waive as noise" is not available. What remains open is Step 1's choice between waiving on the stated mechanism (hot layer-0 experts moving between the 6900 XT tail and the heads in a 1,000,000-swap stress that the live policy never produces) and scoring the placement a warmed policy settles on (Step 2: freeze the warmed placement, then score it). Raw run: run folder policy-kld, out-aa.

Tooling note: since S1/S2 the benchmark's server-config path needs `RADIANCE_HOME` and `BC_SERVE_BINARY` in the environment, and `engine.radiance_home` is a required bc-serve key; the older run scripts under runs/policy-kld do not set them.

## Change Log

- 2026-11-10T13:12:09.846895+00:00 (created-by): Created by agent
- 2026-11-11T10:22:01.303225+00:00 (updated-by): Updated: section:notes
