---
id: BCOP17
order: 17
plan: bc-optimizations
state: pending
created-at: '2026-10-05T04:46:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# Gate whole-expert parallelism behind lower-complexity expert caching

## Description

Backfill of the earlier MET04 audit. Before implementing whole-expert parallelism, compare static tiering and upstream-style GPU expert caching at equal VRAM. Expert parallelism proceeds only if a measured service/transfer bottleneck remains.

## Steps

1. Qualify cache budgets 0/5/10/20% with cold/warm and forced miss/hit controls on gfx1100/gfx1201.
2. Measure TG plus pp512/2048/8192, cache hit/miss, upload bytes/time, evictions and displaced resident layers.
3. Use MET01 as placement/profile owner and existing scheduler cache mechanism where possible.
4. Proceed to EP only if cache/static tiering leaves a quantified bottleneck that EP can address.
5. Keep MET05 auxiliary-device execution separate from primary-device EP/cache policy.

## Related

MET01, MET04, MET05; llama.cpp #29887.

## Acceptance Criteria

- Equal-VRAM cache-vs-static measurements exist.
- EP has a quantified residual bottleneck or is explicitly parked.
- No second cache/placement/router implementation is created.
