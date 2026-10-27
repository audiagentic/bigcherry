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

2026-10-27 RESULT (harness tools/lab/radiance/rad25-event-wake.cpp, runs rad25-event-wake-10271530 and later, branch feat/rad25-blocking-completion): at decode cadence (100/150/300 us between completions) every blocking variant keeps the lane thread at ~100% CPU, same as the yield loop; no variant starves the queue. Per-event hipEventBlockingSync is ignored (block = 100% at every period); only device-wide hipDeviceScheduleBlockingSync sleeps, and only after a fixed ~0.3-0.4 ms spin per wait in ROCr (lane CPU 61/33/18/10% at 0.5/1/2/4 ms periods). ROC_ACTIVE_WAIT_TIMEOUT=0 changes nothing (the spin is ROCr's, not ROCclr's); HSA_ENABLE_MWAITX keeps CPU time at 100% (cannot judge its power from CPU time; needs package power, PWR01). Conclusion: blocking completion events save no CPU in decode on ROCm 7.2; not adopted. Revisit only with a ROCr whose blocked-wait spin is configurable/shorter, or measure MWAITX package power under PWR01.

## Change Log

- 2026-10-27T02:15:19.709939+00:00 (created-by): Created by agent
- 2026-10-27T04:44:03.072985+00:00 (updated-by): Updated: section:notes
