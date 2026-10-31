---
id: RHP08
order: 8
plan: patching-radiance-heat-placement
state: pending
created-at: '2026-10-31T00:00:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# Runtime tail-heat collector

## Description

Establish the actual route frequency and coverage of the experts the 6900 XT owns, including the MTP layer, from the RHP02 feed. The static profile gave the tail 0.68% of routes; natural text gave 1.7%, random-word prompts 14%.

## Steps

1. Write the protocol, experiment or collector against the code as it is (radiance 1.14.0, remote-MoE protocol v19).
2. Land card-free tests first where the item has any logic; measure only with the gpu lease.

## Detailed Solution & Technical Design

## Code Samples & Guidance

## Files

## Validation

Observed route counts match the engine histograms; the static-profile bias is explained; no second production GPU collector.

## Effort & Risk

Depends on: RHP02. Added by the 2026-10-31 GPT review of this plan.
