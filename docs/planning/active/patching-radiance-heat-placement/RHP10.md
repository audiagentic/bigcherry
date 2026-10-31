---
id: RHP10
order: 10
plan: patching-radiance-heat-placement
state: pending
created-at: '2026-10-31T00:00:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# Long-context thrash experiment

## Description

Test whether expert residency changes explain the 290K-1M decode slowdown (decode 92-122 tok/s to ~290K, 74 at 580K, 50 at 1M). A falling decode rate does not by itself show thrashing: growing KV attention work and memory pressure compete as explanations. Prior evidence: a 6 GiB tail cut expert promotions 8,084 -> 5,767 and host-to-device traffic 51.3 -> 32.2 GiB over a short-context run with no speed change.

## Steps

1. Write the protocol, experiment or collector against the code as it is (radiance 1.14.0, remote-MoE protocol v19).
2. Land card-free tests first where the item has any logic; measure only with the gpu lease.

## Detailed Solution & Technical Design

## Code Samples & Guidance

## Files

## Validation

At 1M context with tails of 0 / 3 / 6 / 14 GiB, with the context filled to ~290K and ~580K: resident expert bytes, misses, mover activity, transfer volume, KV occupancy and decode/prefill throughput; the result decides RHP05-07.

## Effort & Risk

Depends on: none (needs the gpu lease and all four cards). Added by the 2026-10-31 GPT review of this plan.
