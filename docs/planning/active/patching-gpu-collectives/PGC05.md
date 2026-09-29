---
id: PGC05
order: 0
plan: patching-gpu-collectives
state: pending
created-at: '2026-09-29T09:34:02.512413+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P2
work: S
---

# adaptive provider: threshold arg and size-bucket routing table

## Description

Make the adaptive crossover (currently the internal pipeline copy-engine threshold, patch 0840) an argument (--allreduce-switch-bytes), later a size-bucket table (small->host, medium->ccl, large->root). Also move GGML_CUDA_AR_Q8_THRESHOLD to an arg. Depends on PGC04.

## Steps



## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Sweep crossover on dual XTX at several sizes; keep the table form only if the matrix shows different winners per size.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

## Change Log

- 2026-09-29T09:34:02.512413+00:00 (created-by): Created by agent
