---
id: RHP13
order: 40
plan: patching-radiance-heat-placement
state: pending
created-at: '2026-11-01T22:51:59.877467+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: L
---

# Stage 4: one allocator for engine, host and helper residency (managed radiance core/place patch)

## Description

Owner direction 2026-11-01: heat-map allocation must drive placement in all scenarios through one allocator. Today the radiance engine's heat engine (core/place/heat.*, planner.cpp, mover.*, core/engine.cpp tiering_dispatch) moves experts live between its host pool and its own VRAM (one serving run: 2,720 promotions, 215 demotions, ~2,500 slots lent from the KV cache), while the helpers own almost every expert (24,571 of 24,576) statically and the engine skips them (R3_MOE_WORKER_SKIP_OWNED). Stage 4 extends the stage 1-3 placement authority to engine-executable experts and host-backed residency, so experts can move helper->engine and engine->helper live.

## Steps



## Detailed Solution & Technical Design

The allocator distinguishes execution owner, physical residency and the immutable weight source. The engine mover keeps providing local host<->VRAM mechanics; helper movers (RHP06) keep remote host->VRAM uploads; the common allocator decides both, and RHP11 epochs govern execution-site changes. The engine side must allocate weights its original skip-owned plan never loaded, accounting for the KV-cache and activation-arena slot loans. Changes to radiance core/place (incl. bc_moe_remote_place.h) go through a package-only bigcherry patch.

## Code Samples & Guidance



## Files



## Validation

Card-free allocator tests across tiers; replay of moves helper->engine and back with exact routing identity; RHP06-style forced churn across tiers; RHP12 gate on every declared layout.

## Effort & Risk



## Standards



## Acceptance Criteria

Experts move between engine and helpers without restart with correct budget accounting (including KV and arena loans); one allocator decides all tiers; RHP12 gate passes.

## Notes

Depends on RHP11, RHP06 and RHP04 stage 3. Plan source: GPT staged plan 2026-11-01. Supersedes the tiered host-cache part of RHP07.

## Change Log

- 2026-11-01T22:51:59.877467+00:00 (created-by): Created by agent

## Ledger-events

- chg_20261101_225732_planning-a-staged-roadmap-for_4092
- 2026-11-01T22:57:58.493983+00:00 (updated-by): Updated: section:ledger-events
