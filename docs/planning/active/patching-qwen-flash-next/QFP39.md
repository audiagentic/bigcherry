---
id: QFP39
order: 39
plan: patching-qwen-flash-next
state: pending
created-at: '2026-10-07T00:39:52.781091+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P2
work: M
---

# One-shot PCIe P2P AllReduce for small tensors: re-evaluate 1252 on the XTX pair

## Description

External report: GPUs writing slices straight into peers' memory for small tensors replaced RCCL, ~57 -> ~9 us per allreduce. We have 1252_nro03_allreduce_p2p_provider (untested) and 1275_ar_small_latency (evaluated). Re-open with current pin: is P2P available between the two XTXs on this board, and between an XTX and the R9700 on PCIe x4.

## Steps

1. Read 1252/1275 notes and why they stopped. 2. Per-allreduce latency by size on production (1277 size trace). 3. P2P capability probe per card pair. 4. ABBA on 27B dual XTX and Flash-Next.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

1224 reduce correctness probe; per-call latency; decode ABBA.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

## Change Log

- 2026-10-07T00:39:52.781091+00:00 (created-by): Created by agent
