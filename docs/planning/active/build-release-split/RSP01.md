---
id: RSP01
order: 0
plan: build-release-split
state: pending
created-at: '2026-11-16T06:37:56.515652+00:00'
breadth: ''
skill: advanced
created-by: claude
work: L
priority: P1
---

# The radiance release is the engine runtime only; the platform release carries the rest

## Description

Owner direction (2026-11-16): a platform release and a bc-radiance release are both fine, but bc-radiance must be the engine, not build and platform binaries; the platform release may contain the rest.

Today bc-radiance is wider than the engine. The runtime bundle packs the whole build `bin/` (RUN_DIRS in `release/runtime_assets.py`), which holds the engine and also tooling and test binaries (rad-tune, rad-kbench, rad-convert, rad-gbnf-check, rad-tokcheck, rad-schemas, rad-info, r4d_selftest) and, in some builds, the platform's bc-serve. The release also installs from a detached git worktree of the whole-repo tag, and `ci_bundle.BUILD_INPUTS` folds platform tooling (campaign, core, source, patch) into the bundle reuse key.

Goal: the radiance bundle holds exactly the engine runtime; the platform release holds launchers, serve, build and patch tooling; install composes the two.

## Steps

1. Audit: list the contents of a current radiance bundle and classify each file as engine runtime, engine helper, tool/test binary or platform binary.
2. Explicit run-side manifest for the bundle (replace the RUN_DIRS directory copy with a named list of what the engine needs: radiance, plugin libraries and kernels, helpers such as r3_moe_worker, runtime libraries). One test fails on any extra file (tool/test/platform binaries).
3. Reuse key: BUILD_INPUTS limited to what shapes the engine binary (engine patches and components, recipes, build image, the adapter and source materialiser); the test pins this list.
4. Install: compose the platform release (launcher, serving config) with the engine bundle; no git worktree of a whole-repo tag as the engine's installed form. Design note first, implementation after review.
5. Consolidate while here: one place that names the run-side manifest and its consumers (bundle pack, restore verify, install verify); remove duplicated lists and legacy archive names that no release uses (migrate, no shims).

## Detailed Solution & Technical Design

Boundary: engine = files the radiance process loads or execs at run time. Platform = everything that drives it (serve, launchers, config, build, patch, campaign, slice, jobs, release tooling) and diagnostics or benchmark binaries. Tool binaries (rad-*, r4d_selftest) stay available as build outputs for development; they are not release payload.

Open points to settle in the install design note: whether bc-serve ships in the platform release or with a serving component; how an installed host pins platform release + engine release together (a pair recorded in one place).

## Code Samples & Guidance



## Files

release-please-config.json; .release-please-manifest.json; tools/bigcherry/release/install.py; tools/bigcherry/release/ci_bundle.py; .github/workflows/radiance-release-bundle.yml; tools/bigcherry/ package layout; docs/reference describing releases.


## Validation

- Bundle manifest test: pack from a fixture build containing extra tool and platform binaries; the bundle holds only the manifest files and restore verifies exactly them.
- Reuse key unchanged by edits to slice/worktree/jobs/lab tooling; changes when an engine patch, recipe or the build image changes.
- Restore a real current bundle and run the existing release-install smoke; engine launches.
- Full patch/release/core suites and tools/tests/test_*.py pass.

## Effort & Risk

Large blast radius: release-please wiring, install path and CI bundle reuse all change; shared modules may need splitting. Risk: breaking installs from older tags; keep the previous install path for already published tags only if named as an explicit exception, otherwise migrate.


## Standards



## Acceptance Criteria



## Notes

## Change Log

- 2026-11-16T06:37:56.515652+00:00 (created-by): Created by claude
