---
id: PVPS07
order: 0
plan: patching-validation-package-standard
state: pending
created-at: '2026-09-25T23:16:36.887890+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P2
work: M
---

# Production lane: adaptive rounds, pp4096 sentinel, acceptance bound, scoped gating

## Description

GPT review req_4d131e8b7c1c452d on PVPS05: 8 fixed rounds is reasonable only if the CI is decisive; add rounds when inconclusive (pre-declared max, no optional stopping beyond it). Add a pp4096 -sm tensor paired sentinel for prefill-affecting patches. Give MTP draft acceptance an explicit equivalence bound. Gate promotion on the lane only for patches touching TP/all-reduce/device placement/speculative decode (declared in patch.toml subsystems); otherwise record as secondary.

## Steps

1. Record CI width; extend to a declared max_rounds when ci95 straddles -1%.
2. pp4096 paired llama-bench -sm tensor lane.
3. acceptance bound (e.g. |delta| <= 0.02 absolute) as a lane check.
4. Lifecycle gate hook keyed on subsystems.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Unit tests for the round extension rule and acceptance bound; hardware run on 1216 standalone.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

## Change Log

- 2026-09-25T23:16:36.887890+00:00 (created-by): Created by agent
