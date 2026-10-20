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

2026-10-20 progress and decision. Done: the contract header moe_backend.h (formats, sizing, probe; RAD11), the lane kernels split into per-stage headers (RCS02), the worker split into units and its own component (RCS03, RCS04), and the `__constant__ g_shape` device global removed - the lane kernels take the shape as an argument (#334), bit-identical on four real Flash-Next captures (replay_hash, #335). GPT review (session "RCS split review and RCS05 backend plan"): keep the host the owner of streams, allocations, the mailbox protocol and orchestration; one .so per architecture exporting a single versioned C entry point (moe_backend_get_api); prefer a coarse enqueue_experts() plus plan() over a stage-by-stage table so fusion and different scratch layouts need no ABI change; compile kernels as static device code in each plugin (no hipModuleLoad); never create a HIP context inside the plugin; no dlclose until queued work completes; keep template specialisation internal; gfx1030 starts with decode GEMV (wave32, 128/256 threads, 32/64-row tiles, integer-bit-manipulation E4M3 decode, bf16 block scales applied per 128x128 tile, f32 accumulate), prefill GEMM separate, and V_DOT cannot take raw E4M3 codes. Decision: the loadable plugin and the coarse enqueue interface are NOT built yet. enqueue_lane (about 300 lines) is entangled with the lane state, and a second backend's real needs (gfx1030 has no WMMA and no native FP8) are unknown, so freezing an ABI now would be a guess. The seam is cut together with the first gfx1030 kernel in RCS07, using the replay_hash check against the gfx1100 output as the safety net. This item stays pending for that reason.

## Change Log

- 2026-10-18T12:57:46.527170+00:00 (created-by): Created by agent
