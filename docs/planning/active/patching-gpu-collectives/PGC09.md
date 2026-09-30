---
id: PGC09
order: 0
plan: patching-gpu-collectives
state: pending
created-at: '2026-09-30T04:03:45.724752+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P1
work: M
---

# AllReduce: adaptive provider as production default for dual XTX, gated by wire accuracy

## Description

Dual-XTX 27B Q8_0 provider A/B (ab-27b-allreduce, 6 balanced rounds, complete separation): RCCL wins prefill (host pp4096 -8.7%), host pipeline wins decode (tg512 +7.5%, tg2048 +2.5%). 0840 adaptive (host below the size threshold, RCCL above) should take both. The host path compresses to bf16 by default (lossy), so no lossy wire may become a default until it passes accuracy gates.

## Steps

1. Measure 0840 adaptive vs ccl vs host (ab-27b-adaptive, queued).
2. Measure 1272 host f32/bf16/f16/q8_0 (ab-27b-ar-wire, ab-27b-ar-wire-q8, queued).
3. Accuracy gates (tools/lab/ar-accuracy): KLD vs exact RCCL reference on a fixed corpus; MTP acceptance equality across A/B arms.
4. Pick the fastest configuration that passes; if f32 host keeps the decode win, prefer it (no tolerance needed).
5. Make it the production default (0860 --allreduce default / recipe) with contract evidence; tune the adaptive threshold (PGC05).

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Gates (tools/lab/ar-accuracy/gates.py): mean KLD <= 0.001, p99 KLD <= 0.01, same top token >= 99.5% vs RCCL f32 reference (27B Q8_0, 32 x 2048-token chunks); MTP draft acceptance within 0.5 pp across A/B arms. Throughput: balanced A/B with CI95 excluding zero, no regression on prefill vs RCCL beyond noise.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Yardstick: Q8_0 weight quantization itself is ~0.001-0.003 mean KLD vs full precision; the AllReduce wire should add at most ~10% of that. Upstream b11233 already defaults the internal path to bf16 (GGML_CUDA_AR_BF16_THRESHOLD=1).

## Change Log

- 2026-09-30T04:03:45.724752+00:00 (created-by): Created by agent
