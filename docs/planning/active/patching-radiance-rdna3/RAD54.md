---
id: RAD54
order: 0
plan: patching-radiance-rdna3
state: pending
created-at: '2026-11-18T03:35:21.126713+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# Retire fused MoE handoff without inject work on nonconsumer ranks

## Description

Remove provably idle launches on the fused nonconsumer rank while preserving immutable ticket validation, retirement, stream order and fail-closed publication checks.

## Steps

Follow the checked nonconsumer packet in docs/reviews/astra-performance-2026-11-18/IMPLEMENTATION-HANDOFF.md. Obtain protocol review before hardware. Keep the ordinary add, consumer kernel sequence and ticket lifecycle unchanged.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files

engines/radiance/components/moe-remote/moe_mailbox_ops.hip and existing ticket/retirement fixtures

## Validation

Run existing exactly-once and ticket replay checks with negative controls. Once BRBC05 evidence is available, use the narrowly selected leased workload and count launches in a separate trace.

## Effort & Risk



## Standards



## Acceptance Criteria

Same retirements, output and replay behavior across ranked cases. Malformed publication fails closed. Eligible nonconsumer submits fewer operations while the consumer and ordinary add remain equivalent.

## Notes

## Change Log

- 2026-11-18T03:35:21.126713+00:00 (created-by): Created by agent
