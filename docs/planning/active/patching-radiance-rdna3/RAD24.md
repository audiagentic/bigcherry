---
id: RAD24
order: 0
plan: patching-radiance-rdna3
state: pending
created-at: '2026-10-27T02:15:14.508402+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P0
work: M
---

# moe-remote: engine fails fast when a helper process dies (pidfd watcher, notify generation)

## Description

The engine's GPU wait (mailbox_add_wait_kernel) spun on a done word when a helper died without setting helper_fault: an 8.7 h hang after both helper GPUs reset (2026-10-26). PR #427 (stacked on #426): protocol v18 Header::helper_pid/helper_start/notify; engine watcher thread on a pidfd per helper raises helper_fault; every engine CPU wait sleeps on the notify generation; watcher starts before registration and fails closed.

## Steps

1. Done: watcher, notify generation, start-time identity, fail closed, helper startup signal handling (046dc1603, 78c2af4be).
2. Done: kill test, both helpers x idle/decode/prefill, all PASS (run rad20-helper-kill-10271238): request fails in 0.2-5.4 s, engine stays up refusing requests.
3. Pending: idle gate PASS (5.5-7 W); speed A/B vs main running (rad20-liveness-check.sh).
4. Open: startup deadline for a helper that dies before publishing its pid (engine map replay can still wait); test-only gates to kill deterministically during park drain and before pid publish; engine error label (see RAD27).

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files

engines/radiance/components/moe-remote/{moe_helper_watch.h, moe_remote_shm.h, moe_remote_hook.cpp, r3_moe_worker.cpp, worker_lanes.h}; tools/lab/radiance/{rad20-helper-kill.sh, rad20-liveness-check.sh}

## Validation

rad20-helper-kill.sh PASS for both helpers in all scenarios; rad20-liveness-check.sh idle gate PASS and A/B within noise of main/#426.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Split out of RAD20. GPT reviews req_bigcherry_866c58b8ea99 (its parser P0 was incorrect: the loop increments p). Record: docs/research/radiance-history/RAD20-helper-waits-2026-10.md.

## Change Log

- 2026-10-27T02:15:14.508402+00:00 (created-by): Created by agent
