---
id: RHP09
order: 9
plan: patching-radiance-heat-placement
state: pending
created-at: '2026-10-31T00:00:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# Heterogeneous cost calibration

## Description

Measure gfx1100 and gfx1030 execution curves by routed rows and kernel shape, PCIe path costs (the 6900 XT is x4 Gen3; its fixed ~2 ms a large-lane call is probably the activation read, unverified), fixed per-expert overhead, queue interference and shared-link effects.

## Steps

1. Write the protocol, experiment or collector against the code as it is (radiance 1.14.0, remote-MoE protocol v19).
2. Land card-free tests first where the item has any logic; measure only with the gpu lease.

## Detailed Solution & Technical Design

## Code Samples & Guidance

## Files

## Validation

Predicts held-out prefill and decode latency within a documented error band; low-confidence migration is refused.

## Effort & Risk

Depends on: RAD23. Added by the 2026-10-31 GPT review of this plan.
