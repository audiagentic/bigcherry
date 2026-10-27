---
id: RAD29
order: 0
plan: patching-radiance-rdna3
state: pending
created-at: '2026-10-27T02:35:08.802114+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P1
work: S
---

# Engine route compaction: warp-aggregated slot reservation in mailbox_publish_kernel

## Description

mailbox_publish_kernel (moe_mailbox_ops.hip ~177-200) runs one 256-thread block and does a shared atomicAdd(&nr) per accepted route: 40,960 candidates per helper at 4096 rows x top-k 10 (verified 2026-10-27). MORI reserves contiguous spans per warp (intranode.hpp:110-215): ballot the accepted routes, one atomic per wave for popcount, lane prefix rank. Wave32/wave64-correct HIP primitives; uniform loop with an in-bounds predicate so the vote never diverges.

## Steps

1. Variants: B0 current, B1 wave-aggregated atomic, B2 block-wide prefix scan.
2. Kernel microbenchmark on captured route maps, rows 8..4096.
3. Equivalence tests: same route set (order may differ), ownership, no overflow, empty/full/sparse ownership, last partial wave, two helpers.
4. If publish time drops: end-to-end prefill A/B.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files

engines/radiance/components/moe-remote/moe_mailbox_ops.hip; tools/lab/radiance/

## Validation

publish kernel time improves; downstream results identical (helper reduction order checked); prefill A/B not worse; decode unchanged.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

From the DeepEP/MORI-EP brief. Local shared-memory atomic, much cheaper than MORI's remote allocator: expect a modest gain, mostly prefill. No mailbox ABI change.

## Change Log

- 2026-10-27T02:35:08.802114+00:00 (created-by): Created by agent
