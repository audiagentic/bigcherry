---
id: MET04
order: 4
plan: patching-moe-expert-tiering
state: pending
created-at: '2026-10-02T04:45:00.827445+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P2
work: L
---

# 1283 qwen4exp_expert_parallel: whole-expert EP across ROCm0/1/2 + CPU tail

## Description

Replace the tensor-split hot tier with whole experts per device (GPU0/GPU1 hot, GPU2 warm, CPU coldest). Dense/attention/GDN/HC/router/shared expert stay on -sm tensor -ts 3,3,2. Each tier branch computes locally; partials (~10 KiB/layer F32) aggregate on GPU0, then the tensor-split graph resumes. No replication in v1 (add only if measured per-device service time shows a stable hotspot).

## Steps



## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

A/B vs MET03 hot-TP tier on the same placement budget: decode, MTP decode, prefill, per-device expert hits and service time, collective time; greedy parity + KLD.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Rationale: with 10 active experts/token there is enough expert-level parallelism; EP avoids a collective per expert and keeps the slower R9700 off the critical path of every expert. Main risk is cross-device latency, not bandwidth.

## Change Log

- 2026-10-02T04:45:00.827445+00:00 (created-by): Created by agent
