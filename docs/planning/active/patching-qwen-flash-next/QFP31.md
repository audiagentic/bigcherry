---
id: QFP31
order: 31
plan: patching-qwen-flash-next
state: pending
created-at: '2026-10-07T00:39:23.061778+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# MTP draft prompt window: draft head prefills only the last N prompt tokens

## Description

External report (4-GPU Flash-Next build, 2026-10): the MTP draft head prefilling only the last 2048 prompt tokens gave prefill 1130 -> 1249 t/s and decode at depth 46.6 -> 64.9 t/s. Our drafter is the MTP sidecar on ROCm3 (6900 XT); decode slows with depth here too (84 t/s at 8K, 75 at 24K, ~51 at 98K). Hypothesis to measure on Brutus, not an assumed gain.

## Steps

1. Find where the draft context is prefilled alongside the target and what it costs per depth (1317/1318 timing patches). 2. Design a window (env, default = current behaviour) and how the draft KV/positions stay consistent after the window slides. 3. Patch, offline tests, composition check. 4. ABBA at 8K/24K/98K: prefill, decode, acceptance, greedy identity or stated acceptance-only difference.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

ABBA with complete separation at depth, acceptance not lower, target output identical (the draft only proposes).

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Idea source: owner-supplied list from a public post; numbers are theirs on different hardware.

## Change Log

- 2026-10-07T00:39:23.061778+00:00 (created-by): Created by agent
