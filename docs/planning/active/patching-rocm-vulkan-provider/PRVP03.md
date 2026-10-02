---
id: PRVP03
order: 0
plan: patching-rocm-vulkan-provider
state: pending
created-at: '2026-10-02T12:35:07.968893+00:00'
breadth: ''
skill: ''
created-by: agent
priority: P2
work: L
---

# Vulkan meta-backend AllReduce provider

## Description

Expose the meta communication SPI (comm_init / comm_allreduce_tensor / comm_free) from ggml-vulkan via get_proc_address. Phase 0 (patch 1290): default-off f32 host-reduction reference provider proving API wiring/correctness. Phase 1: port BigCherry's HIP host-path design (mapped pinned host staging into every device, single reduce shader, arrival flags, chunked pipelining) without changing the SPI.

## Steps

1. Vulkan get_proc_address exposing the three comm functions.
2. Gate behind BIGCHERRY_VK_ALLREDUCE; unset returns NULL and preserves the stock meta fallback.
3. Phase 0 host-f32 rejects unsupported tensors by returning false (meta fallback takes them).
4. BIGCHERRY_PATCH_HIT only when the provider actually executes.
5. Phase 1: verify ggml-vulkan buffer/command/descriptor lifetimes, then implement the mapped-host data path; do not guess cross-device host-import semantics.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Default path stays stock; env-enabled dual/3-device tensor split fires the marker, passes KLD/output checks; meta fallback still handles unsupported types.

## Effort & Risk

Phase 0 M/low; device/mapped-host reduction L/high (no P2P; Vulkan memory-import/coherency rules are driver-dependent).

## Standards



## Acceptance Criteria



## Notes

## Change Log

- 2026-10-02T12:35:07.968893+00:00 (created-by): Created by agent
