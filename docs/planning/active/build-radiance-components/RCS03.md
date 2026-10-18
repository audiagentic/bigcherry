---
id: RCS03
order: 3
plan: build-radiance-components
state: pending
created-at: '2026-10-18T12:57:38.198287+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P1
work: M
---

# Split r3_moe_helper.cpp (1,513 lines): options, owners, serve loop, harness

## Description

Separate the helper process into: options/env (one place that reads and validates R3_MOE_HELPER_*), owners (heat ranking, select_owned, balance), serve loop (mailbox polling, lane launch), and the harness (capture_request, replay, replay-sweep, trace_dump). The harness moves to the moe-harness component so production builds can leave it out.

## Steps

1. Extract options into one struct built once at startup (replaces scattered getenv calls on the hot path where they are read per call, only where measured safe).
2. Move owners/heat/balance to their own unit with its own offline test (balance_score.py already scores layouts).
3. Move capture/replay/trace into the harness unit behind a build option.
4. Same gate as RCS02 after every step.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Bitwise replay identity; helper starts and serves a short run; production build without the harness unit still serves; patch-lint clean.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Depends on RCS01; can run in parallel with RCS02 if file ownership is disjoint (different files).

## Change Log

- 2026-10-18T12:57:38.198287+00:00 (created-by): Created by agent
