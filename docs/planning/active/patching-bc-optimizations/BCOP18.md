---
id: BCOP18
order: 18
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-05T04:47:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# Qualify topology-aware 6900 XT auxiliary expert execution

## Description

Backfill of the earlier MET05 audit. Distinguish resident-expert activation traffic from dynamic expert-weight transfer: the PCH-attached 6900 XT may be useful when weights stay resident even if it is poor as a miss-driven cache tier.

## Steps

1. Benchmark 4/10/32/128 KiB host-staged and any available direct HIP peer transfer paths independently of RCCL capability.
2. Use persistent pinned buffers, event-driven staging and ping/pong buffering; prohibit per-token allocation/global synchronization.
3. Measure transfer in/out, expert compute and queue delay and score placement as route probability x total service cost.
4. Keep ROCm3 outside the primary Meta/RCCL collective group.
5. Promote whole-layer auxiliary execution before attempting expert-granular placement.

## Related

MET01, MET05, patch 1328.

## Acceptance Criteria

- Small-payload topology measurements exist for the actual 6900 path.
- Direct peer DMA and host staging are independently classified.
- Auxiliary placement is based on measured service cost, not nominal PCIe bandwidth/FLOPS.
- No second collective or placement registry is created.
