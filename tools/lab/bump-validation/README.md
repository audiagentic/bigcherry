# Bump validation matrix: does a pin bump silently break dispatch/build/runtime?

Plan item: RHA12 (pin-bump repeatability; see also PIN_BUMP.md step 5/6)
Status: active
Owner: pin-bump process (standing, run on every future bump)
Question state: infrastructure, not a one-shot investigation

## Question

`patch-rebase-check`/`patch-lint`/`bigcherry check` prove every selected
patch still applies and every static gate still passes after a pin bump --
but none of them ever launch a real server. A bump can pass every static
gate and still silently break real dispatch/build/runtime (a new upstream
CLI flag default, a changed device-enumeration order, a build-flag
incompatibility) that only shows up when a real binary is actually run.
This tool answers, per bump: does a freshly-built binary at the new pin
still launch and complete correctly, alone on each real GPU and together
on the real production multi-GPU topology?

## Inputs

- The current pin's `standard` campaign profile (`config/recipes.toml`
  `[campaign.standard]`), built fresh via the same planner/runner
  `bigcherry build` itself uses -- there is no static `build_id`/`binary`
  to hand-maintain, since both are content-hash identities that only exist
  once a real build has run.
- `tierB-qwen9b-q6k` (models.toml: fits gfx1030/gfx1100/gfx1201 with real
  headroom -- the cross-architecture reference lane) for the four
  single-GPU smoke cells.
- `tierL-qwen27b-q8` (models.toml: the real dual-GPU-only production
  model) for the one dual-XTX multi-GPU cell, run under the real
  `production-dual-xtx` runtime-profile (MTP speculative decode).

## Outputs

`bigcherry runtime-matrix`'s own resolved-cells/status/events under the
`--output` directory this script is given -- 5 cells: one per real GPU
(devices 0-3, `production-safe-single`, `tierB-qwen9b-q6k`) plus one
dual-XTX pair (devices [0,1], `production-dual-xtx`, `tierL-qwen27b-q8`).

## Runtime

GPU required: yes (all 4 real devices; the multi-GPU cell needs 0,1)
Real compilation required: yes (one real build per invocation)
Mutates canonical BigCherry state: no (build + runtime-matrix artifacts
only, same as `bigcherry build`/`bigcherry runtime-matrix` used directly)

## Safety

- Canonical-state mutation: none -- this only drives the existing
  `bigcherry build` planner/runner and the existing `bigcherry
  runtime-matrix` CLI; it adds no new execution path of its own.
- `--dry-run` still performs the real build (build_id/binary cannot be
  faked) but skips launching any server.
- Do not import this script from `bigcherry` production, tests, or
  maintained analysis -- it is a bump-time operator entry point only.

## Disposition

Standing tool, not a one-shot experiment: run this as part of every
future pin bump (PIN_BUMP.md step 5/6) rather than retiring it. If a
cell fails, treat it as a real bump blocker and investigate before the
bump is declared complete.
