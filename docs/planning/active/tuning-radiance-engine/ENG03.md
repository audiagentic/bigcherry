---
id: ENG03
order: 0
plan: tuning-radiance-engine
state: pending
created-at: '2026-11-10T13:11:41.863773+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P2
work: M
---

# Multi-stream decode scaling decomposition: matched 1/2/4/6/8-stream runs with work per step

## Description

One-stream decode steps take ~25 ms and six-stream ~47 ms (engine 340-364 tok/s against ~130): the step grows with batch. Candidates: more distinct experts and helper rows per step, attention, scheduler, drafting. Earlier campaigns (rad23-multi8, rad23-streams2, streams-*, policy-accept, rad49-engine-pcores) did not attribute it on the current build.

## Steps

1. Matched 1/2/4/6/8-stream decode runs, policy on, same prompts; record tokens per step, accepted tokens, engine, helper and scheduler stages per step (lane trace for helpers, per-op profile from ENG01 for the engine).
2. Separate step latency from throughput (more tokens per step at a longer step is not a slowdown).
3. Name the stage whose cost grows with batch.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

A table of per-stage time against stream count, with the policy state and plan key recorded.

## Effort & Risk



## Standards



## Acceptance Criteria

The stage that scales worst with stream count is identified with numbers, and a go/no-go is recorded for a fix.

## Notes

Depends on ENG01 for the engine-side breakdown.

## Change Log

- 2026-11-10T13:11:41.863773+00:00 (created-by): Created by agent
