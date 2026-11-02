# rad0011_plan_memory

**Status:** untested
**Plan item:** RSC03

## Summary

The engine's own memory plan (`Engine::plan()`) as a serving manifest v1 `EnginePlan`, measured or checked, through the
shared v1 library `bc_serving_v1`.

- `radiance … --plan-memory FILE` runs start-up as far as `plan()`, writes what it sized (host consumers the engine pins,
  the checkpoint store's slot, floor and want, one full-length session's KV, the per-card VRAM split including the
  resolved KV budget `kv_resolved`, and the expert plane, each with the engine function it came from) as v1 JSON through
  the generated encoder, and exits before any weight is loaded or the KV cache and arenas are allocated. bc-serve uses it
  as a measured fact, cached by container, engine build, card and planning inputs.
- `radiance … --plan-check FILE --plan-check-floor-ppm N` reads FILE (a plan bc-serve measured) through the shared strict
  reader and, right after the VRAM split and **before any pool is configured**, checks the live plan against it (spec
  section 5.2): stable facts by exact equality (`BC_CFG_PLAN_MISMATCH`, exit 2) and admission on the live split, with
  `kv_required` from the calibrated backing floor `N` (`BC_CFG_ADMISSION_REFUSED`, exit 4). The result is printed once:
  `plan check: admitted` with its margins, or the refusal in the shared `radiance: config: CODE (RAD_E_X, n)` format.
  The two flags go together, `N` is range-checked by the manifest's own validator, and they are interim: the v1
  manifest's `engine_view` replaces them at the slice 6 cutover (RSC08).

## Outcome

untested: package and card-free tests exist; the plan check has not yet run on hardware (a fresh plan, a restart that is
admitted and a one-byte drift that is refused), nor an identity run against main.

## Detail

See below.

## What it does

`core/bc_plan_memory.{h,cpp}` and `core/bc_plan_check.{h,cpp}` (created by the patch) hold the logic. Upstream files gain
only thin calls: `core/main.cpp` one include and one call that takes the plan flags off the command line before rad0009's
manifest reader sees them; `core/engine_bringup.cpp` one include, one call at the end of `plan()`'s VRAM split that
records (and with `--plan-check`, checks) its sizes, and one call after `plan()` in `start()` that writes the plan and
exits; `core/CMakeLists.txt` includes BigCherry's `components/serving/bc_serving_v1.cmake` and links `bc_serving_v1` into
`rad_core`. The build tool passes that file as `-DBC_SERVING_V1_CMAKE` (engine.toml `repo-paths`); without it configure
fails closed. Without the plan flags the calls do nothing.

The query runs the engine's real bring-up up to `plan()`, so it needs the engine card: bc-serve runs it on the serve
path, under the launch's own lease, never from `--explain` or `--dry-run`.
