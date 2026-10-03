---
id: QFP02
order: 0
plan: patching-qwen-flash-next
state: pending
created-at: '2026-10-03T15:20:21.946886+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: S
---

# 1292 QSA k-pool tail truncation

## Description

Patch 1292_kpool_tail_truncate (evaluated, in production profile): truncates the QSA compressed k-pool view to the filled tail instead of the full allocated context, so pooled-key work scales with used context rather than -c.

## Steps

1. Fold GPT review (edge cases near the threshold, empty pools). 2. Contract evidence for promotion.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Greedy identical vs untruncated at 32K/80K (with 1294 for determinism); ms/step ABBA.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Evidence: -14% ms/step at 80K (flashnext ABBA, 2026-10-03). Part of the production candidate measured +26.5% at 80K together with 1291/1294/1297. GPT code review in flight: req_af96db92b8e34cfe.

## Change Log

- 2026-10-03T15:20:21.946886+00:00 (created-by): Created by agent
