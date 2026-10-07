---
id: QFP41
order: 41
plan: patching-qwen-flash-next
state: pending
created-at: '2026-10-07T00:40:00.113369+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P2
work: L
---

# Per-device dispatch threads and other host-thread parallelism: design, then implement behind a flag

## Description

External report: each GPU's kernels and collectives launched by its own host thread instead of one thread for all. Owner position updated 2026-10-07: the threaded option can be revisited, and threads may be used to do work in parallel in other areas where it helps. Earlier (same week) the owner called threading dangerous and asked for it to be done carefully; a threaded per-device layout pass in 1340 was written and reverted then. This item now covers design AND implementation: per-device dispatch in the meta backend first, then other candidates (per-device arena planning/binding, host-side input staging, per-device set_tensor uploads at load). The split shows ~40% idle time that this could recover.

## Steps

1. Idle-time timeline of the split per device (where the single dispatch thread is the limiter). 2. Shared-state inventory: meta buffers, split-state cache, simple-tensor containers, pools, CUDA graph cache, collectives ordering, scheduler callbacks (1326/1336). 3. Design with invariants: what each worker owns, where the joins are, what stays single-threaded. 4. Implement behind a flag, default off: persistent per-device workers, no thread creation per graph. 5. Evidence per use: greedy identity, repeated-run determinism, ABBA, a stress run (long decode + deep prefill).

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Identical greedy text and probes with the flag on; no run-to-run variation over repeated runs; ABBA with complete separation; stress run without hang or crash; ThreadSanitizer build of the host side where practical.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes



## Change Log

- 2026-10-07T00:40:00.113369+00:00 (created-by): Created by agent
- 2026-10-07T00:58:28.837689+00:00 (updated-by): Updated: section:title, priority='P2', section:description, section:steps, section:validation
