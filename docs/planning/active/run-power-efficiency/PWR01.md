---
id: PWR01
order: 0
plan: run-power-efficiency
state: pending
created-at: '2026-10-27T02:15:39.545328+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P1
work: S
---

# Whole-system power baseline: every GPU and CPU package at idle, decode and prefill (incl. R9700 engine idle)

## Description

Only the XTX helpers' idle power has been gated (RAD20, 6-7 W). Measure where the watts go across the serving host: board power of each GPU (amd-smi) and CPU package energy (RAPL/amd_energy) at idle, single-stream decode and long prefill, in the production Flash-Next layout. Includes the R9700 engine's idle power, never measured; if high, check the same two causes as the XTX (pending AQL wait, more than num_cp_queues compute queues).

## Steps

1. Extend burst-idle.sh (or a sibling) to sample every inventory GPU plus CPU package energy per phase; devices from env settings.
2. One run; record watts per device per phase and joules per token for decode/prefill.
3. If the R9700 idles high, profile its queues/waits like RAD20.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Per-device, per-phase table recorded on the model page and in the research record.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Measurement only, no speed risk. RX 6900 XT runs an unrelated service; record it but out of scope for tuning.

## Change Log

- 2026-10-27T02:15:39.545328+00:00 (created-by): Created by agent
