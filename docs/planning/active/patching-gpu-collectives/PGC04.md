---
id: PGC04
order: 0
plan: patching-gpu-collectives
state: pending
created-at: '2026-09-29T09:33:59.429712+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# 0860 allreduce provider CLI (--allreduce / --allreduce-wire)

## Description

Base-framework patch 0860_allreduce_provider_cli. Replace env GGML_CUDA_ALLREDUCE with --allreduce {auto,ccl,host,adaptive,p2p,root} and orthogonal --allreduce-wire {native,q8}; unknown name or unsupported provider+wire combo is a startup error. Old names (nccl/internal/hybrid) and GGML_CUDA_AR_WIRE removed; all callers/docs/tests migrated in one change (no shims). Emits BIGCHERRY_PATCH_HIT naming the selected provider. Updates requires/conflicts for 0840, 1244, 1250, 1252.

## Steps

1. GPT authors package (req_2db947dd40ee4492 + amendment req_1a26da4733a148d2). 2. Apply, patch-lint, hardware-free tests under tools/tests/patch. 3. Migrate docs/reference/architecture/MULTI_GPU_DISPATCH.md and scripts. 4. Dual-XTX matrix: auto/ccl/host/adaptive (p2p/root/q8 only as fail-closed unless supported).

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Hardware-free anchor tests; then order-balanced ab-benchmark per provider on GPUs 0,1 (Qwen3.8-27B Q8_0, -sm tensor, MTP n_max=4), activation marker per arm.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Framework change: no promotion gate. Q8 wire is lossy activation compression, orthogonal to model weight quant.

## Change Log

- 2026-09-29T09:33:59.429712+00:00 (created-by): Created by agent

## Ledger-events

- chg_20260929_135722_allreduce-methods-are-now-sele_7144
- 2026-09-29T13:57:26.139352+00:00 (updated-by): Updated: section:ledger-events
