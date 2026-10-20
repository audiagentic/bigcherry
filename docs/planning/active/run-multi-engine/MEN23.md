---
id: MEN23
order: 23
plan: run-multi-engine
state: pending
created-at: '2026-10-18T14:40:58.891273+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P1
work: S
---

# accept runs the helper-ab entry on a build without the hand-off it needs

## Description

First real radiance acceptance run (1.8.0, 2026-10-19): accept took RAD_BIN/RAD_HOME from the pin-bump lane build (radiance:stock, serving-core only). The helper-ab config needs the rad0002 hand-off (R3_MOE_WORKER_SKIP_OWNED) and the moe-remote-handoff experiment build; on the stock build the engine tries to hold every expert and startup stops 'out of memory' (the KLD entry passed). pin-bump --accept would hit the same. Each [[acceptance]] entry must declare the lane and experiment it runs on (engine.toml), and accept (and pin-bump --accept) must build or locate that build itself instead of reading RAD_BIN from the environment.

## Steps

1. Add lane and experiment fields to [[acceptance]] in engine.toml; validate at declaration (exit 2).
2. accept builds or finds the entry's lane build and exports RAD_BIN/RAD_HOME itself; the environment variables stay only as an explicit override.
3. pin-bump --accept uses the same path.
4. Offline tests with a fake lane; document in docs/reference/engines/radiance/README.md.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

A fresh checkout runs `accept` end to end with no manually set RAD_BIN/RAD_HOME; the 1.8.0 helper-ab entry passes on the hand-off build.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Found by the 1.8.0 acceptance run; worked around by building the hand-off experiment lane and exporting RAD_BIN/RAD_HOME by hand.

## Change Log

- 2026-10-18T14:40:58.891273+00:00 (created-by): Created by agent
