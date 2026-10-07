---
id: QFP33
order: 33
plan: patching-qwen-flash-next
state: pending
created-at: '2026-10-07T00:39:30.097678+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P2
work: S
---

# Draft ubatch cap: smaller draft compute buffer

## Description

External report: capping the draft's ubatch cut its compute buffer 457 -> 247 MiB. Check our draft compute buffer on ROCm3 and on any target card it touches, and whether a cap changes draft speed or acceptance.

## Steps

1. Read draft compute buffer size from a production load. 2. Find the existing knob or add one. 3. Measure memory, decode, acceptance.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Memory report before/after; decode ABBA; acceptance unchanged.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

## Change Log

- 2026-10-07T00:39:30.097678+00:00 (created-by): Created by agent
