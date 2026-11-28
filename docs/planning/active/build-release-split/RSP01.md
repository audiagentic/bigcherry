---
id: RSP01
order: 0
plan: build-release-split
state: in_progress
created-at: '2026-11-16T06:37:56.515652+00:00'
breadth: ''
skill: advanced
created-by: claude
work: L
priority: P1
---

# The radiance release is one Python-free engine runtime

## Description

Owner decision (2026-11-17): the `bc-radiance` release is **one engine runtime asset** consisting of matched `bin/radiance` and `radiance_home/bin/bc-serve`, helpers, plugins, runtime libraries and RSC12 per-model configs. `bc-serve` owns start/stop, tuning, guards, power, host preflight, resources, calibration and heat placement in C++. The engine's `archs.toml` is compiled into it. **No Python and no shell launcher** are shipped in the engine release.

The `bc-platform` release is separate, for repository-side build, patch, campaign, slice and release tooling. It is not installed, paired or required to run an engine. Phase 1 scopes the binary bundle but still relies on a detached tag worktree and repository-side launcher; phase 2 designs a verified, repository-free engine install after the RSC02/RSC12/RSC04 cutovers. `rad-*` and `r4d_selftest` are development outputs, not engine payload.

## Steps

1. Audit: list the contents of a current radiance bundle and classify each file as engine runtime, engine helper, tool/test binary or platform binary.
2. Explicit run-side manifest for the bundle (replace the RUN_DIRS directory copy with a named list of what the engine needs: radiance, plugin libraries and kernels, helpers such as r3_moe_worker, matching bc-serve, runtime libraries). One test fails on any extra file (tool/test/platform binaries).
3. Reuse key: BUILD_INPUTS limited to what shapes the engine binary (engine patches and components, recipes, build image, the adapter and source materialiser); the test pins this list.
4. Install: verify one tag-bound engine asset and run it through native `bc-serve` with no repository worktree, Python or shell launcher. Design note first; implement after RSC02/RSC12/RSC04 acceptance.
5. Consolidate while here: one place that names the run-side manifest and its consumers (bundle pack, restore verify, install verify); remove duplicated lists and legacy archive names that no release uses (migrate, no shims).

## Detailed Solution & Technical Design

Boundary: engine = `bin/radiance`, matching `bc-serve`, helpers, runtime plugins/libraries/catalogs, compiled architecture defaults and packaged RSC12 configs. Platform = **repository-side** build, patch, campaign, slice and release tooling, not a paired runtime. The existing shell/Python launcher is interim development tooling and is never packaged. `rad-*` and `r4d_selftest` are excluded.

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

### Phase 2: single Python-free engine release (implementation in progress)

**Historical phase-1 baseline; superseded by the milestone plan below.** Phase 1 delivers an engine-scoped archive, but `release-install <tag|latest>` currently creates a detached whole-repository worktree and the lab launcher still uses shell and Python to start that engine. RSC02 step 1 is an **engine-only** C++ session owner, not yet the fully autonomous helper/tuning launch path. This section is a future contract, not an existing installer interface.

#### One release asset and the engine boundary

Publish exactly **one** `bc-radiance-<version>` engine asset for the caller's selected engine tag, with **binaries and data only**. No Python package, Python executable/script, shell launcher or shell library belongs to this asset. The separately tagged `bc-platform-*` release remains the platform's build, patch, campaign, slice and release tooling; it is **not needed to launch or tune an installed engine**. There is **no** platform runtime archive, compatibility pair/matrix, two-tag selector or paired receipt.

| Future engine asset contents | Reason and ownership |
|---|---|
| `bin/radiance` and `radiance_home/bin/bc-serve` | Exactly matching engine and native C++ serving/session-owner binaries, always from the same build. |
| `radiance_home/helpers/r3_moe_worker`, helper `.so` files, kernel/architecture/quantizer plugins, `kernels.rkb`, optional `kernels_moe.rkb`, vectors | Engine-coupled helper, plugin, kernel and catalog files required by the supported models. Exclude `rad-*` diagnostics and `r4d_selftest`. |
| `runtime-prefix/lib*/` shared libraries and `bigcherry-build-metadata-radiance.json` | Runtime dependencies and build/SDK metadata, not build trees, tests, lab tools or caches. |
| Versioned `configs/flashnext.toml`, `configs/qwen27b.toml` and their complete required data closure | Readable RSC12 per-model configs/variants, shipped **with the engine**, not with the platform. Local site/model/power/heat inputs remain external. Replace the old `engines/radiance/layouts/*.toml` at cutover; do not ship both spellings. |
| Architecture and native defaults **embedded in `bc-serve`** | `engines/radiance/archs.toml` is an **engine build input** compiled into the C++ executable by `engines/radiance/components/serving/CMakeLists.txt` (`archs_seed.inc`). `helper_profile.toml` and `calibration.toml` are also embedded there. Do not independently package a mutable `archs.toml` or let a platform version override it. |

`tools/bigcherry/release/runtime_assets.py:RUN_MANIFEST` currently covers the phase-1 binaries, runtime libraries, catalogs and build metadata, **not** RSC12 configs; its authoritative allowlist, per-file hashes, restored-file verification and CI reuse identity must be expanded/tested when configs become release inputs. Changes to the compiled `archs.toml`, engine sources, native defaults or bundle payload contract must invalidate inappropriate reuse. Verify config discovery and all runtime dependencies in an extracted prefix with **no** repository checkout or Python/shell import path.

RSC02 defines `bc-serve start CONFIG` (detached), `bc-serve serve CONFIG` (foreground owner), `bc-serve stop`, `bc-serve status` and `bc-serve start --manifest FILE` (stored manifest with fresh admission). On leased lab hosts the GPU lease wraps **foreground `serve` for the entire session**, not `start` alone. Today's `engines/radiance/components/serving/main.cpp` exposes `serve/start --run-dir ... [--configs ...]`, not the final named-config or stored-manifest UX. The Python supervisor and shell launcher are behavioral references only and will be retired (RSC02/RSC04), never shipped.

The host, not the release, supplies GPUs, ROCm SDK/driver, models, site paths/settings, optional route heat, calibration and live-policy data. These are mutable **inputs** to resolve, fingerprint and preflight, not private assets to copy. Anything needed to run and optimally tune a released engine—config resolution, memory/host resource planning, host checks, power/gfxoff, calibration and heat placement—must ultimately run in C++ inside `bc-serve`. Until a behavior has been ported, it is a **repo-side lab tool only** and the affected standalone mode is unsupported rather than silently invoking Python.

#### RSC02, RSC12 and RSC04 dependency gates

Compare [RSC02](../run-serving-configuration/RSC02.md), [RSC12](../run-serving-configuration/RSC12.md) and [RSC04](../../completed/run-serving-configuration/RSC04.md). A C++ parser/probe already existing does **not** imply its full start-time workflow has been ported.

| Gate | Verified implementation and remaining work |
|---|---|
| RSC02 **step 1: native session owner** | `engines/radiance/components/serving/main.cpp`, `session.cpp` and `session.h` implement dense/engine-only measured start, child ownership, health, canary, control socket, stop and native SDK refusal. `session.h` explicitly defers helper/power/lab paths. `tools/bigcherry/serving/supervisor.py` is the behavior reference, **not** payload. Complete the intended named-config interface and acceptance. |
| RSC02 **step 2: helpers and resources derive-launch** | **Not yet in the C++ session owner:** helper-first spawn/readiness, helper memory residency, measuring with helpers up, host-pool/resource derivation and failure cleanup. Current production helper runs use `tools/lab/radiance/launch-flashnext-best.sh` and `python3 -m bigcherry resources derive-launch`. Port this before shipping helper-layout startup. |
| RSC12 **config variants and stored manifests**; RSC02 step 5 | `engines/radiance/configs/flashnext.toml`, `qwen27b.toml` and `components/serving/config_loader.cpp` exist, but strict variant/parity acceptance, named-config launch, shipped config closure and `start --manifest FILE` with static-key verification plus fresh live admission are **not yet the release path**. Finish RSC12 steps 4–6 and RSC02 step 5 before retiring layouts. |
| RSC02 **step 3: guards and lab modes** | **Not in the native session lifecycle:** MTP replay guard, live placement/heat recording, lab calibration workflow and its watchdog/ABBA prompts. `components/serving/calibration.cpp` and live-policy code are native building blocks, not full launch orchestration. The current scripts remain `tools/lab/radiance/launch-flashnext-best.sh` and `tools/lab/radiance/calibration_watchdog.py`. |
| RSC02 **step 4: power and gfxoff** | **Not in the native owner:** all-or-nothing `gpu-power apply/restore`, gfxoff floor/check and restore only after verified child exit. Current dependencies: `tools/bigcherry/cli/power_cmds.py` and launcher commands `bigcherry gpu-power` / `bigcherry gfxoff`. Meet PWR09 requirements in C++, no shipped Python fallback. |
| Host preflight and heat-placement utilities | `components/serving/sdk_check.cpp` already handles native SDK checks; the remaining host warnings/checks from `tools/bigcherry/cli/host_check.py` and launcher `lib/sdk-check.sh` must be included in native admission. `tools/lab/radiance/heat-placement/heat-watch.py` is a **repo-only** tuning watcher, not a released utility. Native config/heat control must cover any required optimizations. |
| RSC02 **step 6** and RSC04 **retirement** | Hardware parity and RHP12 on dense and helper configurations, caller/installer migration and deletion of the shell launcher and `lib/plan-check.sh` follow the native conversion. `tools/bigcherry/release/install.py` still makes a tagged source worktree; it is not yet the standalone installer. |

**Interim:** tuned helper layouts launch through the **repository** `tools/lab/radiance/launch-flashnext-best.sh`, the Python `bigcherry` CLI and development layout files. This is a development toolchain **never shipped in an engine asset**. Missing C++ functionality must refuse/declare the corresponding standalone feature unsupported until its gate lands.

#### One tag, one verified install receipt, retained rollback

The caller supplies **one exact Radiance tag**. The release record at **that tag** supplies the permitted asset name and **archive SHA-256**. The installer verifies downloaded bytes against that **tag-bound digest**, then checks product, scope, manifest digest, all file digests/modes, provenance, matched `radiance`/`bc-serve` and host SDK. No signing, signed attestation, pinned index, second tag or compatibility matrix is required. The digest is integrity relative to the selected tag's release record, not an independent publisher-identity assertion.

A proposed **single-engine** install receipt stores `tag`, `asset_name`, `archive_sha256` (from this tag's release record), `manifest_sha256`, engine/`bc-serve` hashes, source/build identity, `rocm_version`, installed prefix and timestamp. It has **no** platform object, compatibility data, pair ID or launcher re-exec field. Verify into a safely staged immutable engine prefix, publish the receipt **last** atomically and reverify installed files/receipt on launch or rollback; invalid paths, mixed assets, changed digests and interrupted installs refuse. Engine source identity may use the selected tag's Git objects rather than a detached whole-repository checkout.

Retain the **current** and **immediately previous** verified engine release for rollback and verify the previous candidate again before switching. **Never convert existing installs**, and never rewrite legacy v1 receipts. A fresh reinstall from a tag can replace **other older** installs according to the current/previous policy, but must not destroy an active or unverified rollback target. The sole previously published unscoped asset (`bc-radiance-1170.0.0`, `radiance-e5c7cb3dcc920259656edb17ec57ffc0.tar.gz`) keeps only the phase-1 immutable SHA-pinned `PUBLISHED_LEGACY_ASSETS` exception, never automatic promotion to the new scoped standalone format.

#### Implementation slices and refusal gates

1. Add RSC12 configs/data to the **engine** manifest, not a platform packer. Assert that the exact extracted runtime file set contains no Python/shell launcher or development tools and independently resolves configs from outside the checkout; test missing/extra/tampered files, metadata, embedded architecture and reuse invalidation.
2. Finish RSC02 steps 2–5 and RSC12 parity/stored-manifest checks. Port helper resources, live/power/gfxoff/host-check/calibration/heat behavior to C++, with synthetic partial-start, stale-key, host/SDK failure, helper death, power-restore refusal and tag-digest mismatch tests.
3. Complete RSC02 step 6/RSC04 acceptance on dense **and** helper hardware, migrate callers, retire shell/Python start surfaces, then implement the one-tag worktree-free installer and current/previous rollback. Test existing v1 and pinned unscoped legacy installs **unchanged**, interrupted staging, unsafe paths and rollback. Run card-free component, release, core and patch suites; use a separate leased GPU acceptance gate.

### Phase 2 implementation milestones (2026-11-18)

**PR 1 — plan and audit.** This plan, its dependency inventory and typed
failure contracts are the reviewable first milestone. Verify release
code, workflow, build metadata, RSC12 configs, and native serving owner.
Validation: no runtime path changed; inspect and commit the plan.

**PR 2 — asset layout and exclusion gate.** Extend the single
`tools/bigcherry/release/runtime_assets.py:RUN_MANIFEST` to include exactly
`configs/flashnext.toml` and `configs/qwen27b.toml` from the *tagged engine
source* (not site/model/heat data) in the archive root, along with matched
`bin/radiance`, `radiance_home/bin/bc-serve`, helper/plugin/catalog/runtime
libraries, build metadata. Add `radiance_home/bin/bc-release` once built.
Make no-Python/no-shell policy a mandatory pack, extraction, restore and
pre-upload check: reject .py, .pyc, __pycache__, interpreter shebangs,
text references invoking an interpreter, and shell scripts/launchers.
Retain the authoritative allowlist, refuse extras and missing requirements,
and ensure reuse keys include new asset contract and config content via
engine_tree. Files: `runtime_assets.py`, `ci_bundle.py`,
`tools/tests/release/test_bu01_runtime_assets.py`,
`test_ci_bundle.py`, `docs/reference/build/RELEASES.md`. Validate
archive payload/layout, extra/missing/tampered members, config digests,
strict source identity and no interpreter dependency in offline fixtures.

**PR 3 — native installer/verifier.** Add a C++20
`radiance_home/bin/bc-release` executable from
`engines/radiance/components/serving`: `release_install.{h,cpp}`,
`release_cli.cpp`, CMake target/installation and
`component.toml [test] targets`. For one exact `bc-radiance-*` tag, parse
its bound release record and permitted asset name/SHA-256, stream-verify
the archive and manifest, reject unknown paths, duplicate/unsafe/nonregular
members, file hash/mode or source/build identity mismatches, and SDK
incompatibility. Extract to isolated staging, publish immutable versioned
prefix and single-engine receipt last, then atomically set current/previous.
Expose `install`, `verify`, `status`, `rollback`. Reverify the rollback
candidate before switching; refuse corrupt current/previous states and
interrupted staging; never rewrite phase-1 receipts or auto-promote the
one pinned historical unscoped archive. Tests: `release_install_test.cpp`
as a **real** card-free CTest declared in component.toml with two
tiny tagged archives, install/verify/rollback/tamper/wrong-tag/traversal/
failure-injection cases, and ASan+UBSan/g++/clang++ strict. The CLI must
function outside the source checkout with no Python executable or shell.

**PR 4 — release workflow cutover.** Update
`.github/workflows/radiance-build.yml` and
`radiance-release-bundle.yml` to package the exact native asset, run the
no-Python gate against both newly built and reused archives before upload,
publish tag-bound checksum/receipt and exclude runtime-source checkouts.
The repository Python packing/building/patch/campaign/slice/release code
belongs only to the independent `bc-platform` release and is not shipped.
Update `docs/reference/build/RELEASES.md`, migrate new-format callers,
delete dead new-format compatibility surfaces; preserve only the named
published legacy archive exception. Validate complete release/core/patch
Python suites (platform-side), patch-lint, forbidden-terms,
clang-format, clang-tidy positive hunks and strict/sanitized C++.

**Final acceptance (other lane)**: RSC02 step 6/RSC04 hardware parity
and launcher retirement and RSC12 named config parity. Step-5 review fixes
merged in #1069. On a leased host prove healthy dense and helper
`bc-serve serve` from the extracted prefix, live guards/power/heat,
stop/status, installed receipt recheck and rollback. No GPU or lease in
PRs 1–4. Never claim the new asset is production-ready until these gates land.

**Implementation ledger, 2026-11-18 (phase 2):** PR #1072 merged the
plan and dependency inventory; PR #1077 merged the exact Python-free
engine payload and two committed RSC12 configs; PR #1079 merged the
native `bc-release` C++20 install/verify/rollback, current/previous
retention, atomic receipt/index, card-free native CTest and real production
PAX/gzip writer golden. The workflow cutover is the fourth slice: new and
reused engine archives both pass a mandatory final publish gate against the
exact selected Git tag, byte and file digests, config closure, no Python/shell
or interpreter references, and build-input/runtime identity. The release
channel still publishes a single archive plus its checksum and tag-bound
receipt; these three files describe one engine asset, not a platform pair.
CI/platform Python code is a build/publish dependency only, never packaged.

**Remaining phase-2 dependencies — owned by RSC02 step 6/RSC04, not
by the release workflow slice:**
1. Remove the legacy `launch-flashnext-best.sh` and `lib/*.sh` runtime
   launcher paths and their remaining callers from the release/start
   distribution; the asset already rejects shell files, but the repository
   development launcher is still maintained during migration.
2. Prove native `bc-serve serve CONFIG` and detached `start CONFIG` from
   the installed tag's `configs/*.toml`, with stored-manifest freshness,
   helper/power/heat/calibration/health and SDK guards. Assert there is
   no source checkout, Python or shell in that live startup call graph;
   finish RSC12 parity and RSC04 mode acceptance on leased dense/helper
   hardware. Tests are prepared card-free but cannot substitute for
   leased production acceptance.
3. Remove the legacy Python `tools/bigcherry/release/install.py`
   `release-install` implementation and any compatibility re-exec path
   only after the native start/install cutover; preserve *existing*
   phase-1 installations and the single pinned published legacy archive
   without conversion. Do not add a shim.
These remain explicit refusal gates, not assertions of production readiness.

#### Python/shell dependency inventory and native replacement

| Release install or startup dependency remaining in the old path | Native replacement: requirements-derived design, not a Python port |
|---|---|
| `release/install.py` Python install, Git tag resolution, source worktree creation and `release-install latest` | `bc-release install` binds an exact supplied tag to that tag's archive digest/receipt and installs without a source checkout. Any future `latest` transport discovery must reduce to a concrete tag before verification; no mutable index is a trust root. |
| `release/runtime_assets.py` Python verifier/extractor, shell `sha256sum` | `release_install.cpp` bounded regular-file archive parser, streaming SHA-256, exact manifest file map and typed refusals. Python packer remains CI/platform-side, never payload. |
| Worktree/build selection, current link/receipt writing, manual rollback | Native `bc-release install/verify/rollback` sealed prefix/index with atomic last-write receipt, current and previous candidate validation before switching; refuse an unsafe state rather than reinterpret legacy receipts. |
| `tools/lab/radiance/launch-flashnext-best.sh` and `lib/plan-check.sh` | `bc-serve serve/start` through `session.cpp`, `manifest_writer.cpp`, `plan_cache.cpp`; RSC02 measured-plan and durable-admission contracts, not copied shell branches. |
| `tools/bigcherry/serving/supervisor.py`; shell health/stop/status | Native `bc-serve` session ownership, control socket, health/canary, stop/status; RSC02 typed process lifecycle. |
| `python3 -m bigcherry resources derive-launch`, repository layouts, CLI config resolution | Native `config_loader.cpp`, `resources.cpp`, stored manifest gate; `configs/*.toml` shipped, `archs.toml` compiled in. RSC12 schema and live measured-memory requirements are the source. |
| `bigcherry gpu-power`, `gfxoff`, Python host checks, `lib/sdk-check.sh` | `power_transaction.cpp`, `power_linux.cpp`, `sdk_check.cpp`, `runtime_guards.cpp`; PWR09 durable admission/restore invariants and refusal-first host checks. |
| `calibration_watchdog.py`, `heat-watch.py`, calibration/heat shell steps | `calibration_lab.cpp`, `live_policy*.cpp`, `runtime_guards.cpp`; RSC02 S8 validity, real producer shapes and heat lifecycle, no runtime Python. |
| Repository Python build/patch/campaign/slice/release scripts and actions | Stay in the separately tagged `bc-platform` tooling; never in the engine asset or runtime installation/start dependency graph. |

**Refusal gate matrix:** exact-tag mismatch, missing/forged tag record,
bad archive digest, wrong product/scope, manifest digest drift,
missing/extra/duplicate/traversal/symlink/non-regular file, unwanted
Python/shell/interpreter reference, tampered bytes/mode, mismatched
engine / bc-serve provenance, SDK mismatch, corrupt receipt, broken
current/previous target, staging crash, and unverified rollback
must all fail closed with a typed native error.

**Scope/gate update (2026-11-18):** RSC02 steps 1–5 are merged on main,
and step-5 review fixes merged in #1069. RSC12 source configs and
native loader exist; the old source layouts remain until RSC02 step 6
and RSC04 acceptance. All hardware and retiring the legacy launcher/
supervisor are explicitly out of this lane. Standalone helper startup
must not silently import Python when unsupported.

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
- **Phase 2 validation gate (not phase 1):** install one tag-bound Python-free
  engine asset with packaged RSC12 configs and run native `bc-serve` from an
  extracted prefix, both for dense and helper configurations.
- Full patch/release/core suites and `tools/tests/test_*.py` pass.

## Effort & Risk

Large blast radius: native launch behavior, engine config closure, install path and CI bundle reuse. Preserve old installs without conversion, retain current/previous rollback and keep the one explicit published unscoped archive exception.

## Standards



## Acceptance Criteria

- The future engine payload contains no Python or shell launcher, includes engine binaries/data and RSC12 configs, and compiles `archs.toml` into `bc-serve`.
- One exact tag-bound archive digest and single-engine receipt suffice; no platform runtime, compatibility pair, signing or migration is assumed.
- Explicit RSC02/RSC12/RSC04 gaps, current/previous rollback, legacy allowance and refusal-oriented tests gate implementation; phase 2 is not claimed as shipped.

## Notes

Progress (2026-11-17): phase 1 merged in #1016 with engine-scoped binary payload, matched `bc-serve`, integrity checks and a pinned legacy allowance. Owner revised phase 2: one Python-free engine release with RSC12 configs, native RSC02 startup/tuning and no platform pair; old installs are not converted. Standalone installer/launch still pending RSC02/RSC12/RSC04 gates.

Progress (2026-11-17): phase 1 merged (#1016). Phase 2 redesigned (#1022, replaces the pair design of #1019): one Python-free engine asset of binaries and data, start-up through bc-serve, tag-bound digest trust, current and previous release kept, gate table mapping each remaining Python or shell behaviour to its RSC02 step. Rewrite lanes for RSC02 step 2 (helpers and resource derivation) and step 4 (power transaction) are in review in C++ (a rewrite from requirements, the Python only a golden oracle); step 3, step 5 and the step 6 hardware A/B and retirement follow. Phase 2 build work is blocked until those merge.

2026-11-25 power parity follow-up: verify installed native bundle can resolve authorized named power policy from existing host/site source and registry with same owner, no Python runtime or private BDF in release. Add card-free installed fresh/replay test and link actual leased nonstock apply/restore evidence from PWR10/RSC02. Existing generated v3 contract and native YAML remain unchanged, no separate config format.

2026-11-25 owner direction: no launch manifest, all authored serving configuration YAML, no compatibility shims or legacy backports. docs/design/radiance-native-yaml-launch.md supersedes earlier manifest-launch instructions; completed RUM04/05/06/09/12 evidence is historical, not proof of the new launch path. Existing capture manifests may remain output-only. No new programme or second resolver. Installed release acceptance: one native --config engine path and same YAML helper path, no active --manifest launch CLI/reader or old writer child in binary, no duplicated site/profile TOML or private host data. Existing capture output is evidence only. Full S18 current supported config matrix; routine release Gate E plus affected configs, dense/helper parity, actual native qualified min-cap foreground and restore with RSC11/PWR10. No duplicate release lane.

2026-11-25 cross-programme bump review: docs/design/radiance-bump-integration.md is the shared implementation-order and compatibility specification; MEN24 owns sequence and timing, domain implementation remains here. No duplicate checker, bump driver, manifest launch or legacy fallback. Installed release gate verifies exact candidate bundle/existing bundle verification, plain plugin name/version/upstream release, row ABI/layout and selected overlay precedence, native YAML roster and owner/helper --config (no full launch reader), exact independent power policy receipt and safe restore. Integrate MEN17 candidate-bound required acceptance into existing release verification/pin-release; skipped or build-only not qualified, deliberate promotion remains. No second release lane or private site YAML bundled.

2026-11-25 second integration review: docs/design/radiance-integration-review-20261125.md is the exact-source review and controls conflicting earlier wording. Shipping pin remains b6b8751 (2.2.0), upstream observed main 1c104de4 is a development candidate, 119 commits and 327 changed files; no automatic repin or backport. No new compatibility layer, launch manifest, bump driver or identity machinery. Gate E #1559 records 2020.11.5 stock archive two layouts with rad0065 unselected; #1554 subsequent promotion not proven by that canary. Retain existing identical-runtime carry rule, native full supported cutover matrix distinct from routine two production layout + affected tests. Existing upstream rad_plugin_abi_layout required alongside version even when integer unchanged; native loader exports and plain plugin versions, no additional digest format. Verify installed actual hierarchy/row/coverage and exact release identity.

2026-11-25 serving/configuration closure sweep: docs/design/radiance-serve-config-review-20261125.md. Retain approved asset allowlist and installed membership checks: runtime_assets.NATIVE_RELEASE_PROFILES and release_archive.cpp kNativeProfiles currently duplicated (18) with test_native_profiles.py equality and generated presets. Derive from one existing approved list/build input, dynamic size and exact installed membership; never allow arbitrary archive YAML. Ship one YAML policy parser and portable profiles, not private site. Gate E script no --power-profile override; verify actual native launch, status/readback/stop, no old executable reader; historical evidence readers not prohibited.

Owner scope correction 2026-11-25: docs/design/radiance-native-yaml-launch.md section Scope and acceptance rule governs. Preserve behavior; smallest native launch and YAML policy change, no generalized machinery. Reuse existing tests/evidence, one shared dense/helper installed acceptance plus materially different affected paths only; cheap parser checks across approved YAMLs. Independent outstanding work stays with its owner, not a blanket S18 prerequisite. Keep current approved asset allowlists/equality test and Gate E; no generator refactor or full GPU roster matrix. Reuse RSC11 and MEN17 same installed evidence, add only affected paths.

2026-11-25 lean build sweep at 79c61a758. One existing CMake/Ninja build graph, one canonical matched release lane, existing tests and release records. No new cache, build system, manifest, hash scheme or generic framework. Reuse actual evidence; only test changed behavior and dependency closure. BU03 #1571 cold versus shipped five IQR resource records: new release compilation must use current tagged source and matched SDK/recipe with cache disabled until actual discrepancy explained/fixed, not a second verification build. Keep existing identical-runtime verified archive reuse, normal runtime asset inventory, ABI/layout and Gate E. ci_bundle.py BUILD_INPUTS includes bc-ccache but not campaign/ccache_paths.py (stable path/ccache options used by workers); review whether output-affecting changes are captured by existing build key and update existing list/oracle only where necessary, no new key scheme. Match actual tag/build and package same tested artifacts, no repack/rebuild of a different tree. No mandatory full release GPU test for docs-only or unchanged verified runtime.

2026-11-25 implementation: runtime_bundle rejects duplicate declared component destinations and ambiguous basename matches instead of picking first sorted match. Uses existing declared outputs and build dirs; still single-match discovery, no new output registry. Existing mock-runtime test proves duplicate and ambiguous refusal. Production R3_ONLY reset and temporary release-only ccache bypass as BU03. 41 selected tests passed; full hosted CI and actual release remain unverified.

PR #1574 first hosted integrated build reached ngram-helper packaging then rejected its intentional root-level library plus POST_BUILD runtime copy. Corrected the helper CMake target to emit directly under RAD_PLUGIN_OUT/helpers and select only the existing runtime-home or explicit component-relative output locations, not recursive basenames; duplicate declared destinations and two valid candidate locations still refuse. 75 offline tests passed including updated bump-gate fixtures and an old helper basename fixture. Actual ROCm CMake configuration generated direct helper and host plugin link targets. Full corrected CI required before merge.

RV4754: only qualified baseline release after exact candidate, required acceptance, install/rollback and source audit. No broader fleet/shared capacity/power changes as implicit promotion requirements.

## Change Log

- 2026-11-18 (RSP01 phase-2 release cutover): merged #1072/#1077/#1079;
  single Python-free asset + native exact-tag install, verify and rollback
  delivered. Final pre-upload gate checks built/reused archives against
  selected-tag content and runtime identity. Deferred: step-6 launcher
  retirement, installed `bc-serve serve CONFIG` acceptance and deletion
  of the Python `release-install` path; hardware gate remains open.

- 2026-11-18 (phase 2): in_progress; four PR milestones, native rewrite boundaries, all known runtime Python/shell dependencies, refusal and hardware acceptance gates.

- 2026-11-16T06:37:56.515652+00:00 (created-by): Created by claude
- 2026-11-16T19:22:34.283744+00:00 (updated-by): Updated: section:notes
- 2026-11-17T03:19:19.707415+00:00 (updated-by): Updated: section:notes

## Reviews

- RV4517
- RV4520
- 2026-11-22T10:47:57.333866+00:00 (updated-by): Updated: section:detailed_solution
- RV4553
- 2026-11-22T12:18:04.798791+00:00 (updated-by): Updated: section:detailed_solution
- RV4586
- 2026-11-25T01:07:16.136319+00:00 (updated-by): Updated: section:notes
- RV4598
- 2026-11-25T01:40:33.632790+00:00 (updated-by): Updated: section:notes
- RV4618
- 2026-11-25T02:11:00.306859+00:00 (updated-by): Updated: section:notes
- RV4653
- 2026-11-25T02:51:20.561788+00:00 (updated-by): Updated: section:notes
- 2026-11-25T02:53:44.093549+00:00 (updated-by): Updated: section:notes
- RV4669
- 2026-11-25T03:32:26.110400+00:00 (updated-by): Updated: section:notes
- 2026-11-25T04:30:56.056571+00:00 (updated-by): Updated: section:notes
- 2026-11-25T05:21:19.232802+00:00 (updated-by): Updated: section:notes
- 2026-11-25T05:33:01.183505+00:00 (updated-by): Updated: section:notes
- 2026-11-25T08:40:51.708103+00:00 (updated-by): Updated: section:notes
- 2026-11-27T15:14:40.634706+00:00 (updated-by): Updated: section:notes
