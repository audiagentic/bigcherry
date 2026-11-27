---
id: RSC18
order: 0
plan: run-serving-configuration
state: pending
created-at: '2026-11-27T03:42:08.275428+00:00'
breadth: ''
skill: advanced
created-by: agent
work: L
priority: P1
---

# Shared KV live accounting, mixed-length admission and scheduler progress

## Description

RSC16 stage C owner, depends on RSC17 capacity contract. Reuse KVManager and Scheduler; code guide E-F. No second scheduler, cache allocator or per-client maximum reservation.

## Steps

1. Test unchanged scheduler with RSC17 shared capacity. 2. Fix proven demand/accounting/progress gaps only; reuse existing allocator and prefill/recompute guards. 3. Measure rectangular block tables; no ABI redesign without evidence. 4. Extend existing metrics and tests for exact pages, cancellation, fairness, overflow, exhaustion, speculation and deferred release. 5. Hand off to RSC16.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Existing mem/kv_window/sched/checkpoint/server and engine e2e tests; shared prefixes, mixed lengths, exhaustion, speculative rollback, deferred release, bounded preemption, no lost state. Compare actual committed/used/free/evictable page ledgers, including error injection.

## Effort & Risk



## Standards



## Acceptance Criteria

No max-window reservation per client; existing length/slot guards; exact per-group page accounting and safe failure; progress under contention, no live eviction/truncation; unchanged legacy path; runtime metrics reflect actual capacity. RSC16 owns GPU mixed-length and cross-family release acceptance.

## Notes

RSC16 second design review/code guide H: existing Scheduler::init/add_locked/fresh_prefill_can_fit already use actual context/prompt demand. Audit first, make only proven corrections. can_grow vs ensure runtime-window geometry diverges; verify or share read-only demand helper, preserve COW and elastic semantics. Explicit context overflow vs transient pressure, no silent truncation in new contract. Keep rad0036/rad0028 dependencies.

## Change Log

- 2026-11-27T03:42:08.275428+00:00 (created-by): Created by agent
- 2026-11-27T03:47:57.369602+00:00 (updated-by): Updated: section:notes, section:steps
- 2026-11-27T03:49:13.270930+00:00 (updated-by): Updated: section:notes
