---
id: RCS07
order: 7
plan: build-radiance-components
state: pending
created-at: '2026-10-18T12:57:54.768641+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P3
work: L
---

# RDNA2 (gfx1030) MoE backend: capability probe and kernel plan

## Description

RDNA2 has no WMMA and no FP8, so the gfx1100 FP16/AHALF paths do not carry over. Write the gfx1030 backend against moe_backend.h: weights stay in the same W4 layout; activations quantised to int8 (not FP8) with dot4/packed-int8 or plain FMA accumulation; wave64/32 and LDS size from the capability bits. Start with the replay harness (correctness vs the CPU reference, per-kernel timing) before any model load.

## Steps

1. Confirm an RDNA2 device is available on a host (none known on Brutus); otherwise stop at the design.
2. Capability probe and backend skeleton that returns a clear unsupported error per lane.
3. Small GEMV lane first (decode), then the grouped lane.
4. Per-arch settings and evidence: nothing validated on gfx1100 is claimed for gfx1030.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Replay agreement with the CPU reference on real captures; then lightweight-tier evidence on the new arch.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Blocked on hardware and RCS05. Activation format differs (int8 vs FP8), so captures need a version bump or a per-backend activation plane.

## Change Log

- 2026-10-18T12:57:54.768641+00:00 (created-by): Created by agent
