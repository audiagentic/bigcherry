---
id: RAD28
order: 0
plan: patching-radiance-rdna3
state: pending
created-at: '2026-10-27T02:35:03.109376+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P1
work: S
---

# moe-remote helper result output: mapped direct writes vs VRAM reduce + staged DMA (selectable)

## Description

The helper's reduction (lane_reduce_tokens_v, moe_lane_reduce.h) writes its whole result straight into host-mapped memory before lane_publish_done. GPU stores over PCIe may lose to a VRAM reduce followed by hipMemcpyAsync on the same stream (MORI AsyncLL separates payload preparation, transfer and completion: low_latency_async.cpp:49-116, 238-329). Add a selectable result method (mapped_direct | staged_dma | auto, auto derived from a measured row-count crossover), default unchanged until measured.

## Steps

1. Replay harness (r3_moe_worker --replay-sweep, real captured requests), rows 1..4096: A0 direct, A1 VRAM + one hipMemcpyAsync, A2 VRAM + chunked copies on the same stream.
2. Record reduce, transfer, publish and total request time, bytes, GPU power, output equality.
3. If a crossover exists: auto picks per row band (calibrated once per device x build).
4. End-to-end A/B (4 rotated pairs) and kill test.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files

engines/radiance/components/moe-remote/{moe_remote_gpu.h, moe_lane_reduce.h}; tools/lab/radiance/

## Validation

Total helper request latency improves on large rows; decode unchanged; output identical; no extra armed queue; kill test PASS.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

From the DeepEP/MORI-EP adaptation brief (2026-10-27). Done is published only after the copy on the same stream. Keep the loser default-off on main if it is not adopted.



BACKLOG TRIAGE 2026-11-04 (revalidated against main 6eb31a708): the proposed disposition "MERGE into RAD41 transport-format study" was NOT applied; this item stays open. Why: engines/radiance/components/moe-remote/moe_remote_gpu.h:1005-1029 shows the production lane reducing straight to the mapped result destination and then publishing completion; the TP-only hipMemcpyAsync at :536 is not a selectable lane output mode. RAD41.md:14-31 studies output formats and sparse bytes, not the staged_dma versus mapped_direct transfer and completion-order A/B of RAD28.md:17-32. Retain the selectable staged-DMA replay and end-to-end experiment, or extend a transfer-mechanism target explicitly before consolidation.

## Change Log

- 2026-10-27T02:35:03.109376+00:00 (created-by): Created by agent
- 2026-11-04T04:24:18.780328+00:00 (updated-by): Updated: section:notes
