---
id: RRVP05
order: 0
plan: run-rocm-vulkan-provider
state: pending
created-at: '2026-10-02T12:35:03.462944+00:00'
breadth: ''
skill: ''
created-by: agent
priority: P1
work: L
---

# Matched HIP versus Vulkan Brutus campaign

## Description

Matched production-style HIP versus Vulkan (RADV, AMDVLK) campaign for Qwen3.8-27B Q8_0 and Qwen3.8-Flash-Next using identical model, split, cache, prompt, seed and probe methodology. Absorbs RRVP04's copy/sync telemetry.

## Steps

1. Arms: HIP production, Vulkan stock RADV, Vulkan stock AMDVLK; later the Vulkan AllReduce subject (PRVP03).
2. Topologies: capacity-gated single-device sanity (27B Q8_0 does not fit one XTX: capacity-skip or a separately labelled partial-offload cell), -sm layer, -sm tensor; dual XTX and the relevant 3-card layouts.
3. Measure prefill, decode, MTP throughput, drafted/accepted counts and acceptance.
4. Fixed-corpus logits/KLD correctness probe before performance acceptance.
5. Capture GPU clocks/utilisation, transfer/AllReduce telemetry and full runtime attestation.
6. For tensor split, separate backend compute loss from collective loss by comparing -sm layer first.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Balanced, order-rotated repeated sessions; correctness/KLD passes before timing is accepted; every cell keeps exact build/runtime/device provenance.

## Effort & Risk

L / medium-high: driver differences and no-P2P communication can dominate results.

## Standards



## Acceptance Criteria



## Notes

## Change Log

- 2026-10-02T12:35:03.462944+00:00 (created-by): Created by agent
