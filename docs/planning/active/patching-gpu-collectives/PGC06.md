---
id: PGC06
order: 0
plan: patching-gpu-collectives
state: pending
created-at: '2026-09-29T09:34:05.557756+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P2
work: S
---

# --allreduce-fuse residual arg (1250 NRO02 fused residual)

## Description

Expose 1250 fused residual (GGML_CUDA_AR_FUSED_RESIDUAL) as --allreduce-fuse {none,residual}; fail closed on unsupported providers. Depends on PGC04. Verify whether 1250 runs over host/ccl or only p2p (p2p is unavailable on the stock kernel).

## Steps



## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Marker patch=1250_nro02 path=allreduce_fused_residual per arm; token parity and paired A/B.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

## Change Log

- 2026-09-29T09:34:05.557756+00:00 (created-by): Created by agent
