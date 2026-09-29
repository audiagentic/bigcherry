---
id: PGC07
order: 0
plan: patching-gpu-collectives
state: pending
created-at: '2026-09-29T09:34:08.605811+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P3
work: M
---

# Additional wire formats (bf16/fp8) on --allreduce-wire

## Description

Evaluate bf16/fp8 payload wire formats as alternatives to q8. New code, not an existing patch; pursue only if q8 shows a measurable gain over native on the dual XTX. Depends on PGC04.

## Steps



## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Accuracy (perplexity / greedy-token parity) plus paired throughput A/B against native and q8.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

## Change Log

- 2026-09-29T09:34:08.605811+00:00 (created-by): Created by agent
