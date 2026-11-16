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

### Phase 2: compatible platform + engine release pair (design only)

**Status: not implemented.** Phase 1 (#1016) supplies an engine-scoped
archive, but there is no platform release archive, pair-compatible receipt,
or worktree-free installer. The following is a proposed contract, not a
description of an existing interface.

#### Current code and runtime file audit

`tools/bigcherry/release/install.py` (`resolve_tag`, `_install`)
selects `bc-radiance-<version>`, creates/validates a clean detached
whole-repository Git worktree at
`$BC_WORK_ROOT/releases/<tag>/src`, verifies its `engine_tree`
and `config_tree`, restores an engine bundle to
`$BC_WORK_ROOT/builds/<source_slice_id>/<plan_id>` and atomically
writes `$BC_WORK_ROOT/releases/<tag>/install.json`.
The v1 receipt contains `tag`, `source` (`record`/`ci`),
`worktree`, `build_dir`, `plan_id`, `source_slice_id`,
`binary_sha256`, `archive_sha256`, `manifest_sha256`,
`engine_tree`, `config_tree`, `rocm_version` and `installed_at`.
`launch-flashnext-best.sh --release` calls `release-install --json`,
then re-execs the launcher *inside that tag worktree* with `--build`.
`runtime_assets.restore` still checks `stock_identity(src)`, although
`runtime_assets.tag_identity(root, tag)` can already inspect source
tree identities from Git objects without a checkout.

Traced inputs on the **platform** side (paths below are repository-relative
today, and should become platform-prefix-relative):

| Consumer | Actual read/execute dependency |
|---|---|
| Shell | `tools/lab/radiance/launch-flashnext-best.sh`; its sourced `lib/sdk-check.sh` (reads build metadata), `lib/plan-check.sh` (invokes engine `bc-serve`) and `lab-env.sh` (selects the lane ROCm SDK). |
| BigCherry Python | `tools/bigcherry/**`, `tools/bigcherry/__main__.py`; launcher invokes `python3 -m bigcherry` for `layout env`, `resources derive-launch`, `host-check`, `gpu-power apply/restore` and `gfxoff floor/check`; the release entry calls `release-install`. The top-level CLI eagerly imports other modules (see `cli/main.py`), so a pruned Python subset has not been proven. |
| Layout and architecture | `engines/radiance/layouts/*.toml` (`flashnext.toml`, `qwen27b.toml`); `engines/radiance/archs.toml`; `engines/radiance/components/moe-remote/topology-v26.schema.json` used by `radiance_layout._wire_schema` for topology predicates. `radiance_layout.layout_sources`/`bc_serve_env` pass the same `BIGCHERRY_LAYOUTS` files to the native resolver; `radiance_inventory.resolve` calls the **engine's** `bc-serve --explain --json` for KFD/PCI role/BDF assignments. |
| Recipe/engine metadata | `engines/radiance/recipes.toml` read by `lab-env.sh` through `campaign.build.lane_sdk_root(..., "linux-radiance")`; `engines/radiance/engine.toml` used for engine descriptors. `core/paths.py` eagerly loads `engines/llamacpp/engine.toml`, an unrelated current Python bootstrap dependency that should be removed or explicitly included until refactored. |
| Setup/lab extras | `tools/env/bigcherry-env.sh` is **operator-sourced**, not sourced by the launcher. `tools/lab/radiance/calibration_watchdog.py` executes only with `--calibrate`. `tools/lab/radiance/heat-placement/heat-watch.py` is suggested in a log message but not invoked automatically. |

`config/environment.local.toml` (gitignored) or
`BIGCHERRY_ENVIRONMENT` carries host-specific paths and power settings;
`config/environment.example.toml` is a template, **not** a usable
production configuration. `BC_WORK_ROOT`, `BC_MODEL_ROOT`, a model
`.rad` container, helper route heat, site/profile/live-policy/experiment
files if selected, live calibration cache, GPU/KFD, ROCm SDK, `/dev/shm`
and external host commands (`python3`, `bash`, `curl`, `flock`,
`agent-lease` when needed) are host prerequisites, not packaged.
The launcher relies on relative `PYTHONPATH=tools` and calibration
script paths; a prefix install must choose a verified working directory
or remove the relative-path assumption.

The **engine bundle** remains authoritative for
`bin/radiance`, `radiance_home/bin/bc-serve`,
`radiance_home/helpers/r3_moe_worker`, kernel/architecture/quantizer
plugins, catalogs, optional vectors and runtime shared libraries, plus
`bigcherry-build-metadata-radiance.json` (`RUN_MANIFEST` in
`tools/bigcherry/release/runtime_assets.py`). The platform must not
ship another `bc-serve`. These are *directly traced* dependencies;
test the full import/config closure in an extracted prefix before
claiming the allowlist is minimal. An existing mismatch also needs
a regression test: `launch-flashnext-best.sh` forwards `--budget`
to `layout env`, while `cli/layout_cmds.py::_resolve` rejects it.

#### Proposed platform payload and release identity

Existing release-please wiring already defines independent components:
`engines/radiance` -> `bc-radiance-<version>` and repository `.` ->
`bc-platform-<version>`. `release-please-config.json` uses separate
release PRs, component-prefixed tags, and excludes `engines` for the
platform line; `.release-please-manifest.json` tracks their **different**
versions. The pair must not be selected by equal version numbers.
Git tags/releases are created only by merged release-please PRs.

Publish a deterministic `bc-platform` runtime asset containing a
**tested, explicit** allowlist of the paths above: launcher, required
shell helpers, importable `tools/bigcherry` package and necessary
engine-descriptor/recipe/layout/topology data. Ship optional lab
calibration support only if included in the declared scope; do not
ship build outputs, diagnostic binaries, engine runtime binaries,
models, private host TOMLs, credentials or mutable caches. For the
first pass prefer the whole Python package to guessing which
eagerly imported modules can be deleted. The manifest
(`bigcherry-platform-runtime-v1`) should bind product/scope,
platform tag and tag commit, platform API, allowed relative filenames
and each file's size/mode/SHA-256, runtime Python/dependency
requirements, supported launcher path and compatibility constraints.
The release receipt pins the archive digest and manifest digest.
Reject archive traversal, symlinks/hardlinks, duplicate/extra/missing
members, changed hashes or executable bits, unsafe parent paths and
unrecognized fields. Treat a matching download checksum as **integrity**
only: authentication needs a trusted tag-bound digest or verified CI
attestation; the source of that trust is an owner decision.

**Versioning hazard:** `engines/radiance/layouts/*.toml`,
`archs.toml` and the topology JSON live beneath `engines/`,
which release-please excludes from `bc-platform` attribution.
`archs.toml` is also embedded into engine `bc-serve` and is a
declared engine build input. Do not copy the latest versions at platform
pack time and call the result immutable. Either move/version the
platform-owned runtime data contract or explicitly connect changes to
both component releases, with digest parity and release-please tests.
Distinguish the engine tag's build-time `config_tree` from the separate
platform tag's runtime config identity; they need not be equal.

#### Compatibility contract and installer algorithm

Extend the *new* engine asset and its `bundle.json` with verified
version/ABI metadata, while retaining phase-1 checks: existing
`schema = bigcherry-radiance-runtime-v1`, `payload_scope =
engine-runtime`, `engine_tree`, build-time `config_tree`,
`plan_id`, `source_slice_id`, `rocm_version`, file hashes,
`manifest_sha256`, engine binary hash and a required matching
`radiance_home/bin/bc-serve`. Add `bc_serve_sha256`, plugin ABI
derived from the real pinned engine/loaded plugins (not a value
borrowed from a test fixture), serving contract identity and
fingerprint, and native topology wire version. The **serving**
manifest is a different schema: generated
`engines/radiance/components/serving/generated/v2/serving-v2.schema.json`
requires `apiVersion = bigcherry/v2`,
`kind = ServingManifest`, `metadata.contract_version = 2`; its
`generated/v2/contract.fingerprint` records schema and semantics
digests. `topology-v26.schema.json` currently has `wire_version = 26`.
The engine binary, matching `bc-serve` and helper must always be from
the same engine asset. No platform-supplied substitute is allowed.

The platform asset declares which engine `product`, archive
`schema`/`payload_scope`, plugin ABI, serving API/contract
fingerprint, topology wire version and layout/architecture input
digests it accepts; include an explicit SDK and Python compatibility
policy. Start with **exact** ABI/schema/fingerprint/wire matches;
version ranges need separate reviewed compatibility evidence. Unknown
or missing compatibility fields refuse the pair, never mean wildcard.

For a selected engine and platform tag, installer must: (1) authenticate
both tag/commit-to-asset identities using Git tag objects plus a
reviewed asset-provenance mechanism; (2) validate both checksum,
archive manifests, per-file SHA-256/modes, product/scope and tag
identity; (3) check the engine's `engine_tree`/`config_tree`
using the **engine tag's Git objects**, not a detached checkout, and
compare the recorded ROCm version with the selected host SDK;
(4) enforce the compatibility matrix and same-bundle `bc-serve`;
(5) stage immutable prefixes outside mutable lab build caches
(e.g. `$BC_WORK_ROOT/releases/objects/engine/<digest>/`
and `.../platform/<digest>/`); (6) atomically publish a verified
pair receipt **last**. No external symlinks, rename replacement,
path traversal, partial receipt or overwrite of differently identified
prefixes. The platform launcher is invoked from the verified platform
prefix with `--build <engine-prefix>`; `BC_SERVE_BINARY` and
`RADIANCE_HOME` derive only from that engine prefix. Never
re-exec a source checkout for an engine release.

**Proposed `install.json` v2 fields** (illustrative, schema to
validate strictly; not an implementation):

```json
{
  "schema": "bigcherry-release-install-v2",
  "installed_at": "<UTC time>",
  "pair_id": "sha256:<canonical pair identity>",
  "platform": {
    "tag": "bc-platform-<version>", "tag_commit": "<git commit>",
    "archive_sha256": "<sha256>", "manifest_sha256": "<sha256>",
    "prefix": "<immutable platform prefix>", "platform_api": 1
  },
  "engine": {
    "tag": "bc-radiance-<version>", "tag_commit": "<git commit>",
    "asset_source": "record-or-ci", "archive_sha256": "<sha256>",
    "manifest_sha256": "<sha256>", "prefix": "<immutable engine prefix>",
    "plan_id": "<32 hex>", "source_slice_id": "<id>",
    "binary_sha256": "<sha256>", "bc_serve_sha256": "<sha256>",
    "engine_tree": "<git tree>", "config_tree": "<build-time git tree>",
    "rocm_version": "<version>"
  },
  "compatibility": {
    "engine_bundle_schema": "bigcherry-radiance-runtime-v1",
    "payload_scope": "engine-runtime", "plugin_abi": "<integer>",
    "serving_api": "bigcherry/v2",
    "serving_contract_fingerprint": "<schema+semantics fingerprint>",
    "topology_wire_version": 26, "layout_inputs_sha256": "<sha256>"
  },
  "launcher": {
    "relative_path": "tools/lab/radiance/launch-flashnext-best.sh",
    "engine_build_prefix": "<same immutable engine prefix>"
  }
}
```

Include platform and engine asset provenance sources, selected SDK
identity and layout manifest identity in the canonical pair ID.
`install.json` is a **claim to reverify** at each install/reuse/launch:
derive allowed prefixes under the configured release store and
recheck file hashes, tags and compatibility; never trust a stored
absolute path without boundary checks. Do not confuse v1
`worktree`/`build_dir` receipt fields with v2 prefixes. A changed
stored receipt, missing file or mixed release must refuse the launch.

**Rollback/migration:** install a replacement pair alongside the active
pair. Verify it fully and complete no-GPU smoke before atomically
switching an active pair *pointer/receipt*; on failure keep the old
pointer, payload and receipt unchanged. Retain old immutable pairs
until rollback no longer references them; reverify the target pair
before rollback. Interrupted staging is never advertised as installed.
Existing v1 `install.json` receipts continue the established tagged
worktree path until **explicit** migration with verification of both
new assets; no silent rewrite, orphaned active build, or automatic
deletion. The single historically published unscoped asset,
`bc-radiance-1170.0.0` /
`radiance-e5c7cb3dcc920259656edb17ec57ffc0.tar.gz`, is accepted
only by the immutable SHA-pinned `PUBLISHED_LEGACY_ASSETS`
v1 exception and must not be eligible for a new scoped v2 pair.
A future CLI might be `release-install --engine-tag ... --platform-tag ...`;
current `release-install <tag|latest>` does **not** choose a
platform tag, so do not silently infer one.

#### First implementation slice and required regression gates

**Smallest safe first PR:** define the `bc-platform` runtime manifest,
packer and verifier plus a hermetic **extracted-prefix** test only.
Do not yet change the active launcher, install interface, v1
`install.json` or installed trees. Use a fake engine binary/`bc-serve`
and fake host/environment/SDK to verify from a non-repository working
directory that platform Python imports, layout files, metadata and
shell paths resolve. Keep the whole `tools/bigcherry` package or first
refactor the eager imports and `core/paths.py`; tests, not intuition,
determine the file allowlist. Assert that neither engine binaries
(including `bc-serve`) nor host-local files are packaged. Reject
missing files, extra files, symlinks, traversal, altered bytes/modes,
wrong product, tag, platform schema or provenance; verify a layout or
topology change changes the versioned compatibility identity.
Test `--budget` forwarding separately before claiming live launch.

Later gated PRs: expose/test engine serving ABI/fingerprint and
cross-asset checks, then replace checkout-based engine identity with
tag Git-object verification, then introduce v2 install/rollback and
launcher switch. Run synthetic good/mismatched pairs (archive SHA,
tag and reused-asset trees, SDK, plugin ABI, serving fingerprint,
topology, layout digest), interrupted/partial pair, symlink/rename
races, missing `bc-serve` or helper, tampered installed asset,
existing v1 unchanged, and legacy-scoped refusal. Run no-GPU install/
CLI smoke, all release/core/patch tests; GPU launch/canary requires
a separate hardware acceptance gate.

#### Open owner decisions

1. Is `bc-platform` a small **serving-only** distribution, or should
   every build/patch/campaign/slice command be operational in it?
2. How are engine-tree layouts, `archs.toml` and topology contracts
   versioned for the platform release without an untracked duplicate?
3. What authenticated source of release-asset identity (tag-bound
   digest, signed attestation/trust root) is required for *both* assets?
4. Must callers explicitly choose both tags, or can `latest` select
   a deterministic pinned compatibility index? What is the public CLI?
5. Which engine/plugin ABI and serving fingerprints are allowed to
   evolve compatibly, and what SDK/Python ranges are supportable?
6. Does calibration/heat tooling belong to the release's platform
   profile? Is fixing `--budget` part of phase-2 launch acceptance?
7. What retention, rollback window and explicit migration policy apply
   to old v1 installs and the one legacy unscoped archive?


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

- Phase-2 design names the code-traced platform runtime input files, external host prerequisites, independent release tags, and the engine-owned binary boundary.
- Proposed versioned manifests and v2 pair receipt bind both authenticated assets, serving/plugin/topology compatibility, SDK, hashes and immutable install prefixes.
- Rollback, v1/legacy migration and an isolated first implementation slice have refusal-oriented tests; owner decisions remain explicit. No phase-2 installer behavior is claimed as available.


## Notes

Progress (2026-11-17): phase 1 merged in #1016: the radiance bundle is the engine runtime only (bin/radiance, radiance_home plugins and helpers, bc-serve which is engine-coupled, runtime libraries); rad-* tools and r4d_selftest are not payload; install verification, reuse key, symlink and legacy-asset guards; three independent reviews. Remaining: phase 2 (the install composes the engine release with a platform release as a compatible pair, no whole-repo worktree for the engine) and the end-to-end launch smoke.

## Change Log

- 2026-11-16T06:37:56.515652+00:00 (created-by): Created by claude
- 2026-11-16T19:22:34.283744+00:00 (updated-by): Updated: section:notes
