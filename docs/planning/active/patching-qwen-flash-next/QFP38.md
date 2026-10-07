---
id: QFP38
order: 38
plan: patching-qwen-flash-next
state: pending
created-at: '2026-10-07T00:39:48.982783+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P2
work: L
---

# Small-batch vector kernels for MTP verify: row packing, wide-K, MoE vector path to 8 tokens

## Description

External report, decode-side kernel work: mmvq small-K row packing on RDNA3 (22.7 -> 25.3 t/s); wide-K mmvq over 8 warps for multi-token batches on 10240x320 projections (~92.6 -> ~96); MoE vector kernel kept for up to 8 tokens so 5-token MTP verify stays off MMQ (80.9 -> 86.7); several rows per warp for short expert down-projection slices (94.2 -> 95.2); 512/1024-thread blocks for tiny long-K F32 matrices (40.6 -> 41.2). Related here: get_mmvq_mmid_max_batch, 1273 (IQ mmvq tuning, evaluated), 1301 (Q8/F32 MTP widths, rejected - read why first).

## Steps

1. Which kernels our 4-token MTP verify batch runs today and their block occupancy. 2. One sub-change at a time, each with its own flag and measurement. 3. Read 1301's rejection before repeating it.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Per sub-change: equivalence test, decode ABBA with acceptance recorded, both architectures.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

## Change Log

- 2026-10-07T00:39:48.982783+00:00 (created-by): Created by agent
