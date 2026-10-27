---
id: RAD30
order: 0
plan: patching-radiance-rdna3
state: pending
created-at: '2026-10-27T02:35:13.767302+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P2
work: M
---

# Engine remote slot mask from publication instead of re-reading Route records (mailbox_route_mark_kernel)

## Description

After helper completion the engine re-reads the Route records from mapped host memory to rebuild the remote slot mask (mailbox_route_mark_kernel, moe_mailbox_ops.hip ~358-393), though the mask is known when publishing. Write a GPU-local mask during mailbox_publish_kernel and use it in mailbox_add_rows_kernel, saving a memset, a kernel and a PCIe re-read per remote layer (decode and prefill). The route-mark kernel also validates sequence, status, counts, bounds, weights and duplicate ownership: those checks must move, not disappear.

## Steps

1. Mode 0 current; mode 1 precomputed mask with the validation kept in a cheaper form.
2. Malformed-result fault tests retained.
3. Benchmark publish->wait->merge per layer and end to end.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files

engines/radiance/components/moe-remote/moe_mailbox_ops.hip

## Validation

Same outputs; every existing protocol check still fires on malformed input; decode/prefill not worse, ideally better.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

From the DeepEP/MORI-EP brief (MORI reuses its routing handle across phases, intranode.hpp:167-257).

## Change Log

- 2026-10-27T02:35:13.767302+00:00 (created-by): Created by agent
