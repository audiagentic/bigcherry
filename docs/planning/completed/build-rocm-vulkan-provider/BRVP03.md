---
id: BRVP03
order: 0
plan: build-rocm-vulkan-provider
state: deferred
created-at: '2026-10-02T12:34:59.291476+00:00'
breadth: ''
skill: ''
created-by: agent
priority: P1
work: S
---

# Queueable dev Vulkan stock build lane

## Description

Promote the existing vulkan-stock source/build/platform from stub status to a queueable stock lane. Build once per source + shader toolchain; RADV/AMDVLK and RDNA2/3/4 are runtime qualification axes.

## Steps

1. queue.sh BUILD rows take an explicit source:build:platform lane (all existing rows migrated; no old-form parser).
2. arch-list '-' means no architecture override (Vulkan).
3. Build llama-server/tools/tests with GGML_VULKAN=ON, GGML_HIP=OFF.
4. Correctness builds with CHECK_RESULTS/VALIDATE separately; timed builds keep both off.
5. Record the glslc cooperative-matrix feature-test results from configure.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Queued vulkan-stock:vulkan-stock:vulkan-linux produces llama-server; the binary enumerates gfx1100/gfx1201/gfx1030 under GGML_VK_VISIBLE_DEVICES; CMakeCache has Vulkan ON / HIP OFF.

## Effort & Risk

S-M / low.

## Standards



## Acceptance Criteria



## Notes

DEFERRED VULKAN PORTFOLIO 2026-11-22: Explicit user-directed Vulkan implementation pause from 2026-09-10 remains in force. Vulkan stock source/build lane is provider-specific and is not required for HIP/Radiance builds. This item is parked as deferred, NOT completed, superseded, deprecated, implemented or promoted; its original source/negative/hardware acceptance remains mandatory. Resume only via Planning API state->pending after a new owner review confirming: Resume only after RRVP02 documents the owner lifting the Vulkan implementation pause; require a pinned Vulkan stock source/build/ICD/driver/SDK receipt using the one existing BuildPlan/BRVP01 build identity, then compile-only and minimal capability tests before any device benchmark. Shared HIP/ROCm identity, SDK, provider capabilities and benchmark controls remain active under their separate owners; this deferral does not block them.

## Change Log

- 2026-10-02T12:34:59.291476+00:00 (created-by): Created by agent

## Reviews

- RV4566
- 2026-11-22T13:05:30.699021+00:00 (updated-by): Updated: section:notes
- 2026-11-22T13:10:30.966959+00:00 (state-transition): State: pending → deferred
