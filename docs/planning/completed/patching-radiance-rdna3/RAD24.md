---
id: RAD24
order: 0
plan: patching-radiance-rdna3
state: completed
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

SIGKILL of either helper at idle, in decode (tokens streaming), in prefill and during a park drain makes the engine fail the request with an error within 60 s (never hangs); the watcher logs the exact pid; idle power and decode/prefill speed unchanged vs the parent build; a helper that dies before publishing its pid cannot strand the engine (launcher ready-line wait + 120 s registration barrier).

## Notes

Split out of RAD20. GPT reviews req_bigcherry_866c58b8ea99 (its parser P0 was incorrect: the loop increments p). Record: docs/research/radiance-history/RAD20-helper-waits-2026-10.md.

2026-10-27: park-drain case added (test-only R3_MOE_WORKER_TEST_STOP_AT=park_drain:<i>); full kill test 8/8 PASS on 63412e63a (idle 0.2 s, decode 5.2 s, prefill 0.2 s, park 5.2-5.6 s, both helpers). Pre-pid death not gated: launcher waits for each helper's ready line (after pid publish) before starting the engine; the 120 s barrier covers a concurrent start. Remaining error-label work is RAD27.

## Change Log

- 2026-10-27T02:15:14.508402+00:00 (created-by): Created by agent
- 2026-10-27T08:22:36.648684+00:00 (updated-by): Updated: section:acceptance_criteria, section:notes
- 2026-10-27T08:22:45.518416+00:00 (state-transition): State: pending → completed
