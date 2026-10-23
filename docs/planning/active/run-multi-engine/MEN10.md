---
id: MEN10
order: 0
plan: run-multi-engine
state: pending
created-at: '2026-10-14T22:16:07.786755+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: L
---

# Engine-aware pin, bump and patch rebase: radiance bumps like llama.cpp

## Description

radiance is pinned (engines/radiance/recipes.toml pinned = 89cee7c, radiance 1.3.0) and its patches are packages (rad0001-rad0003), but the bump tooling is llama.cpp-only: pin_bump.py resolves ProjectContext(engine="llamacpp"), and repin, pull, audit, patch-rebase-check, pin-status and patch-disposition take no --engine. So radiance cannot be bumped and its patches rebase-checked the way llama.cpp's are. radiance also has a second surface llama.cpp lacks: BigCherry plugin components (libr3, libr3moe, the helper) built against radiance's plugin ABI (rad_plugin_abi_version) and internal headers, which a bump can break without any patch anchor failing.

## Steps

1. Add --engine (default from context, explicit for radiance) to repin, pull, audit, patch-rebase-check, pin-status, patch-disposition and pin-bump; resolve upstream, vendor, patches and recipes from engines/<name>/engine.toml.
2. Declare radiance's required trees in its recipes ([[trees]]) - dev build tree.
3. Bump gate for components: rebuild every component against the new revision, check rad_plugin_abi_version and the component offline tests; a component break STOPs like a patch break.
4. Update the bump-llamacpp skill into an engine-generic bump skill (or add bump-radiance) covering the component gate.
5. Fire-test: bump radiance to the latest upstream release (MEN12).

## Detailed Solution & Technical Design

Inventory 2026-10-15 on main 63cc952f: 106 hard-coded llama.cpp engine references in 34 files (cli/patch.py 23, release/pin_bump.py 17, patch/evidence.py 8, campaign/qualification_matrix.py 8, patch/source.py 7, patch/gates.py 4, the rest 1-3).

Design:
1. One global `--engine <name>` on the top-level parser (beside --llama-root), required whenever more than one engine is declared and the command is engine-scoped; resolved once into ProjectContext.resolve(engine=...) and passed down. No default to llamacpp.
2. Lane/source-driven commands (build, campaign-build, ab-benchmark, experiment) derive the engine from the recipe that declares the source; an explicit --engine that disagrees is an error.
3. Engine-generic commands (patch-*, repin, pull, audit, pin-status, pin-bump, pin-release, release-notes, sources, check, doctor) take the context engine; platform-wide ones (patch-lint, check, doctor) iterate all engines and label output.
4. Commands that are genuinely llama.cpp-only (tuning/replay/autotune, qualification matrix, runtime-matrix, rocprof kernel tools) declare `supported-engines = [llamacpp]` in one table and refuse others with a clear error, instead of silently assuming.
5. Library functions take `engine` as a required argument; no module-level llamacpp constant survives except in engines.py's declaration loader.
6. Delete obsolete one-off scripts (re14_real_run, re15_*, validate_rd_patches, release_validate if unused) rather than migrating them.
7. Bump gate for components (radiance plugins): pin-bump rebuilds components and checks rad_plugin_abi_version.

Order: (a) parser + context plumbing + supported-engines table; (b) patch/* and source/* libraries; (c) cli/patch.py; (d) release/*; (e) delete obsolete scripts; (f) component bump gate; (g) regression tests: each verb resolves radiance paths with --engine radiance, llama.cpp unchanged, llama-only verbs refuse radiance.



## Code Samples & Guidance



## Files



## Validation

Offline tests: each verb resolves radiance paths with --engine radiance and still resolves llama.cpp unchanged (all callers migrated, no default shim). patch-rebase-check --engine radiance --all reports rad0001-rad0003 clean at the current pin. Dry pin-bump to the current pin is a no-op.

## Effort & Risk



## Standards

Engine separation (owner, 2026-10-15: keep each engine's logic and tools separate, not munged together with ifs):
- Generic code never branches on an engine name. What differs between engines is data in engines/<name>/engine.toml (layout, build, serve, prefix) or behaviour behind a per-engine adapter module implementing one small interface (e.g. tools/bigcherry/engines/<name>/ or the existing experiment/engine_quality adapters); generic code calls the adapter.
- llama.cpp-only tooling (tuning/replay/autotune, qualification matrix, runtime-matrix, rocprof kernel tools) stays in its own modules and is gated by one supported-engines table at the CLI boundary, not by checks inside it.
- Allowed comparisons on engine: isolation guards (a patch/module belongs to the engine asked for) and the declaration loader. The one existing behavioural branch (campaign/benchmark.py:809/828, execution evidence for llama.cpp) moves into the adapter.
- Lint for it: a test that fails on a new `== engines.LLAMACPP` / `"llamacpp"` / `"radiance"` literal outside core/engines.py, adapters and tests.



## Acceptance Criteria



## Notes

No back-compat: every caller migrates to the engine-explicit shape. Plan items for radiance work themselves live in patching-radiance-rdna3.

Scope widened 2026-10-15 (owner: 'Is bigcherry now exposing multiple engines via the same cli? Do we need to clean this up'). Today the engine is chosen four inconsistent ways: build lanes imply it through recipe source names (radiance-lab:...); engine-bench takes --engine; patch-lint and the registry walk every engine; and 22 sites hard-code ProjectContext.resolve(engine="llamacpp") (cli/experiment, hardware, jobs, patch, profiling, source, tuning; release/pin_bump; source/sources; patch/validation_campaign; doctor; jobs/worker; validate/re14/re15 scripts), plus patches filter help text naming engines/llamacpp/patches/catalog.toml. Rule to land: every engine-scoped command takes one --engine resolved from engines/<name>/engine.toml (or derives it unambiguously from a lane/source and refuses a mismatch); platform-wide commands (patch-lint, check, doctor) report per engine; no command silently assumes llama.cpp. Migrate every site; delete one-off scripts (re14/re15/validate_rd_patches) if obsolete rather than migrating them.

Progress 2026-10-15: steps a and b in PR #239 (d650725f, 3d06adfb, b4a17b01). Open design points from step b: (1) jobs/worker.py is spawned over SSH with no CLI engine - carry the engine in the job payload and the spawner; cli/jobs.py uses JOBS_ENGINE = engines.LLAMACPP until then. (2) source/audit.check_overlay_sync reads paths.SRC_OVERLAY (llama.cpp's overlay) - take the engine's overlay root. (3) llama.cpp-only campaign modules (patch/campaign/*, validation_campaign) use engines.LLAMACPP + paths.PATCHES; thread the CLI's ProjectContext when they gain a second engine.

Step c (3cb74f3c): patch read commands engine-scoped. Follow-ups: patch_lifecycle.compute_all(engine=...) ignores the engine for plan items (radiance patch-status shows llama plan rows); patch-doc on a missing vendor checkout tracebacks from git instead of a clean error; patch-disposition could be platform-scoped; apply/patch-rebase-check/patch-validate/patch-promote remain llama.cpp tools until the release/bump steps; 18 planning docs keep old invocations as history.

2026-10-15: steps a-d merged (#239, #240, #242); steps e and g in the cleanup PR (60d8e9e3). Paused for quota: step f (component bump gate; per-engine pin-bump compile profile and audit checks - both are ggml-cuda invariants today) and MEN12. Engine-literal allowlist entries still 'threading pending': check.py, doctor.py, patch_admission.py (admit -> gates -> catalog.require_admission -> cli apply), plus cli/jobs.py + jobs/worker.py (engine in job payload) and campaign/benchmark.py's execution-evidence branch (move into an adapter).

## Ledger-events


- chg_20261014_233548_radiance-builds-that-apply-pat_7437
- 2026-10-14T23:35:52.705077+00:00 (updated-by): Updated: section:ledger-events
- 2026-10-15T00:20:08.176517+00:00 (state-transition): State: pending → in_progress
- 2026-10-15T00:20:55.115985+00:00 (updated-by): Updated: section:detailed_solution
- 2026-10-15T00:22:17.544255+00:00 (updated-by): Updated: section:standards
- 2026-10-15T01:30:59.629028+00:00 (updated-by): Updated: section:notes
- 2026-10-15T01:48:20.956679+00:00 (updated-by): Updated: section:notes
- chg_20261015_023051_bigcherry-commands-now-take_7405
- 2026-10-15T02:30:55.727845+00:00 (updated-by): Updated: section:ledger-events
- chg_20261015_023055_pin-bump-and-release-commands_3207
- 2026-10-15T02:30:59.554877+00:00 (updated-by): Updated: section:ledger-events
- 2026-10-15T02:45:47.734006+00:00 (updated-by): Updated: section:notes

- chg_20261017_014810_health-checks-and-the-job-syst_8713
## Change Log

- 2026-10-14T22:16:07.786755+00:00 (created-by): Created by agent
- 2026-10-14T22:29:45.466300+00:00 (updated-by): Updated: section:notes
- 2026-10-17T01:48:14.114861+00:00 (updated-by): Updated: section:ledger-events
