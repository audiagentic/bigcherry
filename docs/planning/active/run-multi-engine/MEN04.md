---
id: MEN04
order: 4
plan: run-multi-engine
state: pending
created-at: '2026-10-08T20:36:21.075596+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P2
work: M
---

# Engine as a declared axis: source, pin and build lane per engine

## Description

config/recipes.toml assumes one upstream (llama.cpp), one vendor tree and CMake/HIP build lanes that produce llama-server. A second engine needs its own identity so that a result always says which engine, which pin and which settings produced it.

## Steps

1. Add an engine field to [source.*] with its own upstream, pin and vendor location; llama.cpp sources state engine explicitly (no implicit default).
2. Container build lane: a lane whose build product is a pinned image digest plus a settings snapshot, not a compiled binary; build identity = digest + settings hash.
3. pin-status and pin-bump report per engine; a bump of one engine never touches the other's trees.
4. Evidence and release records carry the engine.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files

config/recipes.toml, tools/bigcherry/build/builds.py, tools/bigcherry/release/pin_bump.py, tools/bigcherry/core/context.py

## Validation

pin-status lists both engines; a container lane resolves to a digest; existing llama.cpp builds produce the same build plan ids as before the change.

## Effort & Risk



## Standards



## Acceptance Criteria

A lane string selects engine, build and platform; results from the two engines cannot be confused in any record.

## Notes

No legacy path: every existing source gets the explicit field in the same change. Do after MEN03 so the lane has something to run.

## Change Log

- 2026-10-08T20:36:21.075596+00:00 (created-by): Created by agent
