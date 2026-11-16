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

Owner direction (2026-11-17; supersedes the earlier `bc-serve` exclusion):
`bc-radiance` owns the complete, version-matched engine runtime, including
`bin/radiance`, `radiance_home/bin/bc-serve` (built from the Radiance
serving component), `r3_moe_worker`, plugins and catalogs. `bc-serve`
shares engine ABI/manifest and expert ownership; it must not be a platform
binary. Platform owns launcher, layouts and their configuration plus
build/patch/campaign/slice/release tooling. `rad-*` and `r4d_selftest`
remain development products, excluded from engine runtime bundles.

The former bundle copied whole build directories containing diagnostics and
tests as well as engine programs. The scoped manifest packages only required
runtime files. Phase 1 still installs using a detached tag worktree;
phase 2 will bind compatible platform and engine releases without such a
whole-repository engine install.

Goal: immutable, verified engine runtime bundle including `bc-serve`, with
platform tooling and layouts released separately and paired at install time.

## Steps

1. Audit: list the contents of a current radiance bundle and classify each file as engine runtime, engine helper, tool/test binary or platform binary.
2. Explicit run-side manifest for the bundle (replace the RUN_DIRS directory copy with a named list of what the engine needs: radiance, plugin libraries and kernels, helpers such as r3_moe_worker, matching bc-serve, runtime libraries). One test fails on any extra file (tool/test/platform binaries).
3. Reuse key: BUILD_INPUTS limited to what shapes the engine binary (engine patches and components, recipes, build image, the adapter and source materialiser); the test pins this list.
4. Install: compose the platform release (launcher, serving config) with the engine bundle; no git worktree of a whole-repo tag as the engine's installed form. Design note first, implementation after review.
5. Consolidate while here: one place that names the run-side manifest and its consumers (bundle pack, restore verify, install verify); remove duplicated lists and legacy archive names that no release uses (migrate, no shims).

## Detailed Solution & Technical Design

Boundary: engine = the Radiance runtime and tightly coupled serving binary (`bc-serve`), helpers, plugins and catalogs. Platform = launchers, layouts, configuration, build, patch, campaign, slice, jobs and release tooling. Tool binaries (rad-*, r4d_selftest) stay available as build outputs for development; they are not release payload.

### Phase 1: scoped runtime (implemented)

The single `release/runtime_assets.py:RUN_MANIFEST` selects `bin/radiance`,
`radiance_home/bin/bc-serve`, build metadata, the MoE helper, runtime kernel/architecture/quantizer/helper
plugins, kernel catalogs, optional vectors and runtime-prefix shared libraries.
No full `bin/`, `radiance_home/` or `runtime-prefix/` directory is copied.
`payload_scope: engine-runtime` in the manifest and asset receipt identifies
the scope. Scoped restores reject unlisted paths and missing required files;
the receipt lists exact `file_names` and the manifest records per-file
digests; restore checks both agree. CI checks this on pack and restore
and never reuses an earlier unscoped, wider archive. The archived
legacy `radiance-<plan>.tar.gz` in `releases/v1.17.0.json` is still accepted
for that previously published release.

The existing lab launcher checks executable `bin/radiance`,
`radiance_home/bin/bc-serve` and `radiance_home/helpers/r3_moe_worker`,
checks build metadata via `lib/sdk-check.sh`, and selects kernel plugins
from its layout. These build-side files are in the engine bundle; no
additional launch binary has been identified as missing. The tagged
release worktree still supplies the launcher, shell libraries,
`bigcherry` Python layout/resource commands and layout configuration.
Model files, route heat (where needed), a matching ROCm SDK and GPU
hardware are external prerequisites. Thus phase 1 can launch via the
existing platform checkout but the archive is not a self-contained
standalone distribution.

### Phase 2: compatible release-pair install (design ONLY; do not implement yet)

1. **Pin compatible engine and platform releases.** The engine release
   supplies `bin/radiance`, its matching `bc-serve`, helpers and all runtime
   libraries. The platform release supplies launcher, layouts and configuration.
   Verify each release's identity, archive, file digests, SDK and compatibility;
   record both tags, digests, version/ABI constraints and selected launcher/build
   paths in one atomic receipt. Reject a mismatched or conflicting pair.
2. **Remove the whole-repository engine worktree dependency.** Read tagged
   records/identities via git objects and download/install verified engine and
   platform assets into separate immutable prefixes. The platform launcher
   targets the engine via `--build`, deriving `BC_SERVE_BINARY` and
   `RADIANCE_HOME` from the engine prefix; it never carries another
   `bc-serve`. Existing published-tag worktree installs remain until verified
   migration. Validate with synthetic pair/install/upgrade/reject tests and
   a no-GPU install smoke; actual GPU launch acceptance remains separate.

## Code Samples & Guidance



## Files

release-please-config.json; .release-please-manifest.json; tools/bigcherry/release/install.py; tools/bigcherry/release/ci_bundle.py; .github/workflows/radiance-release-bundle.yml; tools/bigcherry/ package layout; docs/reference describing releases.

## Validation

- **Phase 1:** offline bundle manifest test requires `bc-serve` and the
  MoE helper; deleting either refuses pack. Packaging a fixture with extra
  `rad-*`, `r4d_selftest` and platform binaries excludes them, and restore
  verifies the exact manifest membership/digests.
- **Phase 1:** reuse key changes when engine patches, build inputs, image,
  or the payload contract (including `bc-serve`) changes, but not for
  unrelated platform-only edits; reuse verifies the archive file set.
- **Phase 1 launcher audit:** `bin/radiance`, `radiance_home/bin/bc-serve`,
  `radiance_home/helpers/r3_moe_worker`, build metadata and layout-selected
  plugins are all engine-bundle files. The platform launcher, shell libraries,
  Python commands, layout inputs, models and SDK remain external.
- **Phase 2 validation gate (not phase 1):** install and verify an engine
  and platform release as a compatible pair, then perform launch smoke.
- Full patch/release/core suites and `tools/tests/test_*.py` pass.

## Effort & Risk

Large blast radius: release-please wiring, install path and CI bundle reuse all change; shared modules may need splitting. Risk: breaking installs from older tags; keep the previous install path for already published tags only if named as an explicit exception, otherwise migrate.

## Standards



## Acceptance Criteria



## Notes

Progress (2026-11-17): phase 1 merged in #1016: the radiance bundle is the engine runtime only (bin/radiance, radiance_home plugins and helpers, bc-serve which is engine-coupled, runtime libraries); rad-* tools and r4d_selftest are not payload; install verification, reuse key, symlink and legacy-asset guards; three independent reviews. Remaining: phase 2 (the install composes the engine release with a platform release as a compatible pair, no whole-repo worktree for the engine) and the end-to-end launch smoke.

## Change Log

- 2026-11-16T06:37:56.515652+00:00 (created-by): Created by claude
- 2026-11-16T19:22:34.283744+00:00 (updated-by): Updated: section:notes
