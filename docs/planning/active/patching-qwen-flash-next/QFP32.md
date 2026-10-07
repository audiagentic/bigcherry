---
id: QFP32
order: 32
plan: patching-qwen-flash-next
state: pending
created-at: '2026-10-07T00:39:26.596251+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: S
---

# MTP re-reserve: no full realloc + sync per prefill chunk when MTP outputs turn on

## Description

External report: re-reserving the graph once when MTP outputs turn on ended a full GPU realloc+sync per prefill chunk (88.5K prefill 1042 -> 1130 t/s). First step is to check whether we have this cost: count scheduler/allocator reallocations per prefill chunk on production (1339 report, 1340 replan counters, sched reserve logs).

## Steps

1. Measure reallocations per chunk during a deep prefill with and without MTP. 2. If present, reserve for the MTP-output graph shape up front. 3. ABBA prefill at 24K/98K.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Realloc count per chunk goes to zero; prefill ABBA; identical output.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

## Change Log

- 2026-10-07T00:39:26.596251+00:00 (created-by): Created by agent
