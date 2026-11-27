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

Audit ensure/grow_for/recycle_window/COW and rank coordination; reuse fresh_prefill_can_fit and preempted_can_prefill. Unify demand geometry, enforce real page/state and in-flight limits, cancellation and bounded progress. Measure fixed BatchBuilder tables; only redesign if material and ABI proof. Extend existing metrics.

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

## Change Log

- 2026-11-27T03:42:08.275428+00:00 (created-by): Created by agent
