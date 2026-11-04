---
id: MEN04
order: 5
plan: run-multi-engine
state: pending
created-at: '2026-10-08T20:36:21.075596+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P2
work: M
---

# Engine as a declared axis: upstream, pin, vendor tree and build lane per engine

## Description

config/recipes.toml assumes one upstream (llama.cpp), one vendor tree (vendor/llama.cpp, reached through paths.llama_root()) and build lanes that produce llama-server. Radiance is also CMake + HIP, so the existing build machinery (build plans, toolchain resolution, compile check, content-addressed build directories) applies with a different upstream, target binary and CMake options; no container lane is needed for it.

## Steps

1. [engine.<name>] tables: upstream URL, pin, vendor location, server binary, health and shutdown routes. llama.cpp is declared the same way; no implicit default engine.
2. [source.*] names its engine; a lane string resolves engine through its source.
3. paths: engine_root(engine) replaces llama_root(); every caller is migrated in the same change (about 85 call sites in tools/bigcherry, about 200 test files).
4. Build: per-engine CMake options and target; build identity includes the engine; work/builds and work/upstream are partitioned by engine.
5. pin-status and pin-bump take an engine and keep one transition marker per engine, so one engine can be mid-bump while the other is stable.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files

config/recipes.toml, tools/bigcherry/core/paths.py, tools/bigcherry/core/context.py, tools/bigcherry/build/builds.py, tools/bigcherry/release/pin_status.py, tools/bigcherry/release/pin_bump.py, tools/bigcherry/campaign/lane.py

## Validation

pin-status lists both engines; existing llama.cpp lanes produce the same build plan ids as before; a radiance lane builds on dev.

## Effort & Risk



## Standards



## Acceptance Criteria

A lane string selects engine, build and platform, and every result, build and release record names its engine.

## Notes

No legacy path: every existing source gets the explicit field in the same change. Do after MEN03 so the lane has something to run.



BACKLOG TRIAGE 2026-11-04 (SUPERSEDED; revalidated against main 6eb31a708, verdict CONFIRM): engines/radiance/engine.toml and recipes.toml declare upstream, pin, vendor tree and build profile; engine is a declared axis. Evidence: engines/radiance/engine.toml and recipes.toml declare upstream, pin, vendor tree and build profile; engine is a declared axis. Still referenced by: MEN07, MEN08, MEN09. Original content kept below for history.



REOPENED 2026-11-04: an independent revalidation (against main 6eb31a708) rejected the closure recorded above, which had been based on a lighter check. It stays open. Why: engines/radiance/engine.toml:9-28 and campaign/lane.py:702 show declared engine/source/build selection. However MEN04.md:24 requires replacing llama_root() with engine_root() and migrating every caller, while tools/bigcherry/core/paths.py:90-100 still exports llama_root and CLI engine_scope.py:95-103 retains its --llama-root special case.

## Change Log

- 2026-10-08T20:36:21.075596+00:00 (created-by): Created by agent
- 2026-10-08T20:48:02.956123+00:00 (updated-by): Updated: section:title, order=5, section:description, section:steps, section:files, section:validation, section:acceptance_criteria
- 2026-11-04T03:45:59.173838+00:00 (updated-by): Updated: section:notes
- 2026-11-04T03:46:03.350062+00:00 (state-transition): State: pending → superseded
- 2026-11-04T04:43:48.305115+00:00 (updated-by): Updated: section:notes
- 2026-11-04T04:43:56.666192+00:00 (state-transition): State: superseded → pending
