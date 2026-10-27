---
id: RAD25
order: 0
plan: patching-radiance-rdna3
state: pending
created-at: '2026-10-27T02:15:19.709939+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# moe-remote: small-lane blocking completion events instead of the in-burst yield loop

## Description

During a burst each small-lane thread yield-spins on done_sequence (~100% of a core per lane, worker_lanes.h run_small). Replace only the completion observation with hipEventBlockingSync|hipEventDisableTiming events, one per queued slot, recorded after lane_publish_done; keep request gates, queue depth, burst arming, queue count and lane classes unchanged. The historical -14% (feat/rad20-blocking-lanes) also had an idle armed stream wait (2.2x GEMV slowdown, FINDINGS), so it does not condemn events alone.

## Steps

1. Harness first: wake latency (completion -> CPU), CPU per thread, requeue gap, no model load.
2. Implement with slot-generation ownership (never re-record an outstanding event).
3. Stop path: release waits before any blocking sync.
4. A/B vs #427 build (4 rotated pairs, combine on), idle gate, per-thread CPU.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files

engines/radiance/components/moe-remote/{worker_lanes.h, moe_remote_gpu.h}

## Validation

No yield loop in the small lane; per-thread CPU near zero in decode; tg/pp within noise of the parent build; idle gate PASS; kill test PASS.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Split out of RAD20. Reject if decode regresses materially even if CPU drops.

## Change Log

- 2026-10-27T02:15:19.709939+00:00 (created-by): Created by agent
