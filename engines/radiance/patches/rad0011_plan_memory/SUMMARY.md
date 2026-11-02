# rad0011_plan_memory

**Status:** untested
**Plan item:** RSC03

## Summary

`radiance … --plan-memory FILE` runs start-up as far as the engine's own memory plan (`Engine::plan()`), writes what that
plan sized as strict versioned JSON to FILE, and exits before any weight is loaded or the KV cache and arenas are
allocated. bc-serve uses it as a measured fact: the host consumers the engine pins, the checkpoint store's slot and its
floor and want, one full-length session's KV, the per-card VRAM split and the expert plane, each with the engine
function it came from. It is cached by container, engine build, card and planning inputs, so a configuration is
measured once.

## Outcome

untested: package and card-free tests exist; the JSON is not yet compared on hardware against a full start's sizing.

## What it does

`core/bc_plan_memory.{h,cpp}` (created by the patch) hold all the logic. Upstream files gain only thin calls:
`core/main.cpp` one include and one call that takes `--plan-memory FILE` off the command line before rad0009's manifest
reader sees it; `core/engine_bringup.cpp` one include, one call at the end of `plan()`'s VRAM split that records its
sizes, and one call after `plan()` in `start()` that writes the plan and exits. Without `--plan-memory` the calls do
nothing.

The query runs the engine's real bring-up up to `plan()`, so it needs the engine card: bc-serve runs it on the serve
path, under the launch's own lease, never from `--explain` or `--dry-run`.
