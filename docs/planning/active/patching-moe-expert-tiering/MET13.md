---
id: MET13
order: 0
plan: patching-moe-expert-tiering
state: pending
created-at: '2026-10-12T12:21:43.616815+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P2
work: M
---

# Overlapped prefetch of host-resident MoE experts at prefill (Strata-style)

## Description

Owner request 2026-10-12, after reviewing Niko1221/Strata (Qwen3.8-Flash-Next engine for 12-24 GB cards). Strata's long-prefill speed comes from copying whole non-resident expert blobs to the GPU in bulk, a layer/batch ahead, on a separate copy stream into a multi-slot staging ring, overlapped with the current layer's compute (docs/DETAILS.md, src/prefill/prefill.cpp @ 7695859a). Radiance has the same mechanism (core/place/stager.h).

llama.cpp has a basic form: experts placed on the CPU (-ot exps=CPU) are copied to the GPU for large-batch matmuls (op offload), but - to verify against the current pin - not prefetched ahead or overlapped with the previous layer.

Value: today's production Flash-Next layout keeps every expert resident on three cards, so this gains nothing there. It matters when experts must live partly in host RAM: long context with f16/q8_0 KV (the 256K fit only loaded with q4 KV, which the owner rules out), or freeing the R9700.

## Steps

1. Read current-pin ggml op-offload / backend scheduler copy path for MUL_MAT_ID with CPU-resident weights; record whether copies are synchronous, per-op, and on the compute stream.
2. Measure a baseline: Flash-Next with N expert layers on CPU (-ot), prefill 8K/32K, rocprofv3 copy vs compute overlap.
3. Design a patch package: dedicated copy stream + ring of device staging buffers, prefetch layer L+1's expert weights while layer L computes, event-gated reuse; default-off flag during qualification.
4. Mechanics tests, then lightweight-tier evidence (activation marker, ABBA prefill, greedy identity).

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Activation evidence (patch HIT marker / copy-overlap in trace); ABBA prefill at 8K and 32K with experts partly on CPU, complete separation; greedy identity vs patch off; no decode regression.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Strata reference numbers (its README): RX 7900 XTX 24 GB, Orca Flash-Next IQ3_XXS, ROCm 7.1: 447-967 tok/s prefill, 55-59 tok/s decode. Radiance counterpart: README 'Flash-Next: the experts that do not fit are the cost' and the prefill-stager investigation on branch feat/rad11-gfx1100-bootstrap.

## Change Log

- 2026-10-12T12:21:43.616815+00:00 (created-by): Created by agent
