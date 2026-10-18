---
id: RCS05
order: 5
plan: build-radiance-components
state: pending
created-at: '2026-10-18T12:57:46.527170+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: L
---

# moe_backend.h and the gfx1100 backend plugin

## Description

Implement the backend interface from RCS01. The helper process stays arch-neutral; the gfx1100 kernels build into helpers/moe_backend_gfx1100.so, loaded at startup by matching hipDeviceProp gcnArchName. Each device in a helper set may use its own backend, so a mixed-arch box works. Startup prints the backend and its capability bits as the activation marker.

## Steps

1. Define the vtable and capability bits (wmma_f16, fp8_e4m3_act, dot4, lds_bytes, wave_size).
2. Build the existing lane code as the gfx1100 backend with no logic change.
3. Helper: probe devices, dlopen the matching backend, fail clearly on none.
4. Replay harness takes the backend as a parameter so backends are compared on the same captures.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Bitwise replay identity on gfx1100; A/B on Flash-Next shows no regression (call overhead is one indirect call per lane, measured by the trace arm).

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Depends on RCS04. Check the indirect-call cost with the trace overhead arms already in helper-trace-ab.

## Change Log

- 2026-10-18T12:57:46.527170+00:00 (created-by): Created by agent
