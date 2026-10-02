---
id: BRVP03
order: 0
plan: build-rocm-vulkan-provider
state: pending
created-at: '2026-10-02T12:34:59.291476+00:00'
breadth: ''
skill: ''
created-by: agent
priority: P1
work: S
---

# Queueable Brutus Vulkan stock build lane

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

## Change Log

- 2026-10-02T12:34:59.291476+00:00 (created-by): Created by agent
