---
id: QFP41
order: 41
plan: patching-qwen-flash-next
state: pending
created-at: '2026-10-07T00:40:00.113369+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P3
work: L
---

# Per-device dispatch threads: safety design review only

## Description

External report: each GPU's kernels and collectives launched by its own host thread instead of one thread for all. Owner position (2026-10): threading is dangerous to introduce; it may be reviewed only if it can be done safely and helps. This item is a design review, not an implementation: what state is shared (meta buffers, split-state cache, pools, graph cache, collectives), what ordering the collectives need, and what idle time it could recover (the split shows ~40% idle).

## Steps

1. Idle-time timeline of the split per device. 2. Shared-state inventory. 3. Written design with the invariants and a test plan. 4. Owner decision before any code.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Design review accepted by the owner; no code under this item.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

## Change Log

- 2026-10-07T00:40:00.113369+00:00 (created-by): Created by agent
