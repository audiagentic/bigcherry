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
2. Declare radiance's required trees in its recipes ([[trees]]) - Brutus build tree.
3. Bump gate for components: rebuild every component against the new revision, check rad_plugin_abi_version and the component offline tests; a component break STOPs like a patch break.
4. Update the bump-llamacpp skill into an engine-generic bump skill (or add bump-radiance) covering the component gate.
5. Fire-test: bump radiance to the latest upstream release (MEN12).

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Offline tests: each verb resolves radiance paths with --engine radiance and still resolves llama.cpp unchanged (all callers migrated, no default shim). patch-rebase-check --engine radiance --all reports rad0001-rad0003 clean at the current pin. Dry pin-bump to the current pin is a no-op.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

No back-compat: every caller migrates to the engine-explicit shape. Plan items for radiance work themselves live in patching-radiance-rdna3.

Scope widened 2026-10-15 (owner: 'Is bigcherry now exposing multiple engines via the same cli? Do we need to clean this up'). Today the engine is chosen four inconsistent ways: build lanes imply it through recipe source names (radiance-lab:...); engine-bench takes --engine; patch-lint and the registry walk every engine; and 22 sites hard-code ProjectContext.resolve(engine="llamacpp") (cli/experiment, hardware, jobs, patch, profiling, source, tuning; release/pin_bump; source/sources; patch/validation_campaign; doctor; jobs/worker; validate/re14/re15 scripts), plus patches filter help text naming engines/llamacpp/patches/catalog.toml. Rule to land: every engine-scoped command takes one --engine resolved from engines/<name>/engine.toml (or derives it unambiguously from a lane/source and refuses a mismatch); platform-wide commands (patch-lint, check, doctor) report per engine; no command silently assumes llama.cpp. Migrate every site; delete one-off scripts (re14/re15/validate_rd_patches) if obsolete rather than migrating them.

## Change Log

- 2026-10-14T22:16:07.786755+00:00 (created-by): Created by agent
- 2026-10-14T22:29:45.466300+00:00 (updated-by): Updated: section:notes
