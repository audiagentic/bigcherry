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

2026-10-02 first screening (vk-27b-screen, stock Vulkan build b-vk-stock = vulkan-stock:vulkan-stock:vulkan-linux at c061, RADV/Mesa 26.1.3, stock kernel; Qwen3.8-27B Q8_0 ub2048, 1 timed request; greedy identical across all Vulkan layouts). Dual XTX -sm layer: 882 pp / 20.8 tg (HIP 1086 / 23.8). Dual XTX -sm tensor: 494 / 24.2 (HIP 1479 / 38.5) -- meta generic allreduce_fallback (no Vulkan comm provider). Layer + MTP5: 856 / 32.1 (acc 57.8%). Tensor + MTP5: 467 / 49.6 (HIP 1315 / 78.6). 3-card tensor -ts 3,3,2: 286 / 14.8. Single R9700 (-sm none, fits 32 GB): 1120 / 19.2. Reading: Vulkan per-device compute is close to HIP on layer split (-19% pp, -13% tg) but tensor split loses ~2/3 of prefill and ~37% decode to the host-fallback collective -- PRVP03 (1290 phase 0 queued, phase 1 mapped-host) is the gap. AMDVLK not installed on brutus yet.

## Change Log

- 2026-10-02T12:35:03.462944+00:00 (created-by): Created by agent
- 2026-10-02T12:46:59.538869+00:00 (updated-by): Updated: section:notes
