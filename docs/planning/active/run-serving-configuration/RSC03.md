---
id: RSC03
order: 3
plan: run-serving-configuration
state: pending
created-at: '2026-11-02T00:53:06.703369+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P0
work: L
---

# One derived start-up resource plan: host RAM, pinned memory and VRAM budgeted together

## Description

Replace independent memory flags (host pool, n-gram placement, host KV tier, checkpoint slots, helper VRAM budgets) with one resource ledger resolved at start-up from discovered RAM/VRAM/pin and cgroup limits, model geometry and the actual helper owner map, plus a safety reserve and a performance objective (not 'the most that fits'). Measured 2026-11-02 on flashnext-4gpu: the hand-picked 22,528 MiB host pool used ~7.1 GiB; a 9 GiB pool loses nothing fresh (44K prefill 3,825 vs 3,837 tok/s) and the freed RAM as a 12 GiB host KV tier removes the long-running-server slowdown (6-stream decode after an ~1M-token session fill 46.9 ms vs 52.8; 8 sessions cold 3,026 vs ~2,450 tok/s) with the n-gram table kept in RAM; a 32 GiB tier with the n-gram table on disk helped prefill but left 0 GiB free RAM - a derived plan must reject that. Reuse radiance's budget machinery (core/mem/vram_budget.cpp resolve, core/engine.cpp loan_tick, core/mem/kv.cpp spare_tokens/grow_for, core/engine_bringup.cpp resolve_table_placement and size_checkpoint_store); add a managed patch so the engine sizes its host pool after the helper owner map is known (the planner must use the real ownership planner or its published map digest, not a Python approximation).

## Steps



## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Budget arithmetic tests (host, pin cap, cgroup, n-gram, KV tier, checkpoints, expert pool, staging, dual generations); RHP12 fresh and retained-session cohorts pass; configurations leaving no safety reserve rejected with an explanation.

## Effort & Risk



## Standards



## Acceptance Criteria

No fixed memory values in the default launch path; the retained-session cohort matches fresh-server step time within the RHP12 gate; explain shows why each budget has its size.

## Notes

Depends on RSC01/RSC02; feeds RHP13 (runtime) and RCS10 (derived settings). Runtime reclaim between finished-session KV and expert residency stays RHP13.

## Change Log

- 2026-11-02T00:53:06.703369+00:00 (created-by): Created by agent

## Ledger-events

- chg_20261102_005338_planning-one-serve-command_5259
- 2026-11-02T00:53:52.338745+00:00 (updated-by): Updated: section:ledger-events
