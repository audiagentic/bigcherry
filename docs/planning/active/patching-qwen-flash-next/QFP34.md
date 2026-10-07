---
id: QFP34
order: 34
plan: patching-qwen-flash-next
state: pending
created-at: '2026-10-07T00:39:33.618549+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# Thin-F32 prefill kernel: <=16-row F32 matmuls off generic SGEMM

## Description

External report: a dedicated kernel for F32 matmuls with <=16 rows took 401 -> 21 us per call (pp4096 1602 -> 1836 t/s). Use our kernel census to find which F32 matmuls in Flash-Next prefill have this shape and what they cost now.

## Steps

1. Census: F32 MUL_MAT shapes and time share at ub512 prefill. 2. Kernel + dispatch condition, both RDNA3 and RDNA4. 3. Op-level equivalence test vs the generic path. 4. Prefill ABBA.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

test-backend-ops style equivalence within existing tolerance; per-kernel timing; prefill ABBA; greedy identity or documented float-order difference.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

## Change Log

- 2026-10-07T00:39:33.618549+00:00 (created-by): Created by agent
