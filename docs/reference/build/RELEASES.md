# Releases: the formal stages, for every engine

Every engine releases through the same stages. Tags and GitHub releases are made only by release-please from a merged
release PR; nobody tags by hand. Each engine has its own release line (package, changelog, version, tags); the shared
platform has one too.

| Line | release-please package | Component / tag prefix | Changelog |
|---|---|---|---|
| llama.cpp | `engines/llamacpp` | `bc-llamacpp` / `bc-llamacpp-` | `engines/llamacpp/CHANGELOG.md` |
| radiance | `engines/radiance` | `bc-radiance` / `bc-radiance-` | `engines/radiance/CHANGELOG.md` |
| platform | `.` (excludes `engines/`) | `bc-platform` / `bc-platform-` | `docs/releases/PLATFORM_CHANGELOG.md` |

An engine's line is declared in two places, and an engine missing from either is refused by `release-notes` and
`pin-release`: a package in `release-please-config.json` (and its version in `.release-please-manifest.json`), and an
`[engine.<name>]` table in `config/release.toml` (tag prefix, the patch sets that make up the production build, an
older tag family, and extra repository paths owned by the engine). The package path comes from the engine's layout
(`engines/<engine>`).

## Stages

1. **Ledger event.** Substantive work records a change event (`record_change_event`) listing the files it touched.
   Events carry no engine field: an event belongs to an engine's release when any listed file is under
   `engines/<engine>/`, the engine's `engine.toml` layout (patches, overlay, vendor checkout, recipes, components,
   external sources) or its `paths` in `config/release.toml`. An event touching no engine's files belongs to the
   platform line. An event touching two engines appears in both releases.
2. **Release notes.**

       python -m bigcherry --engine <name> release-notes <pin> [--version V] [--stdout]

   llama.cpp notes use the engine's ledger events, patch changes and bump record.
   Radiance notes use a short, reviewed `public_summary` in `releases/<pin>.json`, separate from the internal
   audit and hardware evidence in `notes`. Supply it using `pin-release --summary @file`
   (up to 2500 characters, beginning with a Markdown level-two heading). The summary should cover what changed,
   measured results, checks and known limitations; it must not contain raw gate output. A missing summary stops
   the Radiance release at the record phase.
   Output: `docs/releases/notes/<tag-prefix><version>.md`. Never edit generated notes by hand.
3. **release-please PR.** The notes commit sets the engine package's `release-as` in `release-please-config.json`;
   after it reaches `main`, release-please opens that package's release PR (changelog, manifest version).
4. **Merge and tag.** Merging the release PR makes the tag of record `<tag-prefix><version>` and the GitHub release.
   The release workflow (`.github/workflows/release-please.yml`) then titles the release and attaches the generated
   notes file.
5. **Pin-release record.**

       python -m bigcherry --engine <name> pin-release <pin> --evidence @<file> [--summary @<file>] [--bump minor|patch] [--dry-run]

   runs stages 2 to 4 as resumable phases (`gate`, `record`, `notes`, `main`, `release`, `unpin`, `sync`) and writes the
   gate output and hardware evidence into the pin's release record. `--dry-run` prints the concrete actions and writes
   nothing. See [PIN_BUMP.md](PIN_BUMP.md) for the bump that precedes it.
6. **Runtime bundle (radiance).** CI attaches a verified engine-only bundle to new `bc-radiance-*`
   releases ([How a release gets its bundle](#how-a-release-gets-its-bundle)).
   New `payload_scope: engine-runtime` receipts list the exact selected paths
   (`file_names`) and bind per-file digests through `manifest_sha256` for the engine
   binary, matching `radiance_home/bin/bc-serve`, plugins, kernel catalogs,
   helpers and runtime shared libraries. Diagnostic `rad-*` binaries and
   `r4d_selftest` stay out. Only explicitly recorded published legacy
   unscoped assets retain their historical verification allowance.

## Installing a release

`release-install <tag|latest>` (Radiance) verifies and unpacks the
engine alongside a clean detached source worktree at its tag; see the
[Radiance runbook](../engines/radiance/RUNBOOK.md). **RSP01 phase 1** changes
the bundle payload and strengthens installed-tree integrity checks. It does
not change the `release-install` interface or the tagged-worktree installation
architecture. The engine bundle now contains the three required launch binaries:
`bin/radiance`, `radiance_home/bin/bc-serve`, and
`radiance_home/helpers/r3_moe_worker`, along with build metadata and
the runtime plugins/catalogs. The existing lab launcher can again run
this installed engine; its required build-side binaries are present.

The archive alone is **not** a self-contained launch environment. The
tagged worktree supplies `tools/lab/radiance/launch-flashnext-best.sh`,
its shell libraries, Python `bigcherry` layout/resource commands, and
layouts/configuration. A compatible ROCm SDK, model, hardware and
layout-specific inputs (for example helper-route heat) are also required.
**Phase 2 (design only)** will record the engine and platform (launcher
and layouts/configuration) as a compatible pinned pair, without requiring
a whole-repository worktree for the engine. `bc-serve` stays in the
engine release, not the platform release.

- A `runtime_assets[<tag>]` entry in a pin record committed at the tag is the source of truth and takes precedence
  (receipt `source: "record"`).
- A tag with no entry (a release made by CI) installs from its release's own `bigcherryrad-<plan>.tar.gz`, `.sha256` and
  `bigcherryrad-<plan>.receipt.json` (receipt `source: "ci"`); a receipt or `bundle.json` whose `product` is not
  `bigcherryrad` is refused before anything else is checked. The receipt is the asset entry plus `release_tag`,
  `runtime_key` and an optional `reused_from`. It installs only when the archive matches the `.sha256` asset and the
  receipt; the receipt's `release_tag` is the tag; `runtime_key`'s `engine_tree`/`config_tree` equal the tag's trees
  (git objects at the tag) and the bundle's; every extracted file matches the archive's `bundle.json`; and
  `rocm_version` equals the host SDK. `bundle.json` may name another tag only when `reused_from` names that tag and
  its trees equal this tag's. Any failure refuses the install.
- `latest` is the numerically newest local `bc-radiance-*` tag with a record entry or a release carrying exactly one
  archive with its `.sha256` and receipt. A bundle that then fails verification refuses; it does not fall back to an
  older tag.

### RSP01 phase 2: paired platform and engine releases (proposal, not installed today)

Phase 2 is a **design only**, described in
[the RSP01 plan](../../planning/active/build-release-split/RSP01.md#phase-2-compatible-platform-and-engine-release-pair-design-only).
No `bc-platform` runtime archive, compatibility-pair `install.json`, or
worktree-free `release-install` is available yet. The supported behavior
above remains: `release-install <tag|latest>` selects **one Radiance tag**,
creates or checks a detached **whole-repository** source worktree at
`$BC_WORK_ROOT/releases/<tag>/src` and verifies the restored runtime.
`launch-flashnext-best.sh --release` re-execs that tag's source launcher
with its `--build` directory. Do not treat a `bc-platform-*` tag as a
currently installable artifact.

**Release lines are separate:** release-please owns
`bc-radiance-<version>` (component `engines/radiance`) and
`bc-platform-<version>` (component `.`), using independent manifest
versions and tags. Equal numbers do not imply compatibility; the
platform component excludes `engines/` paths in its current release-please
config, although its launch-time layouts, `archs.toml` and topology data
currently live there. The ownership/bump rule for those files must be
fixed before platform packaging, not assumed.

The proposed platform archive contains a digest-verified, immutable
subset of the repository: `tools/lab/radiance/launch-flashnext-best.sh`,
its `lib/sdk-check.sh` and `lib/plan-check.sh`, `lab-env.sh`, an
importable `tools/bigcherry` package, versioned
`engines/radiance/layouts/*.toml`, `archs.toml`, the topology schema,
and runtime recipe/engine metadata. The launcher currently invokes
`layout env`, `resources derive-launch`, `host-check`,
`gpu-power` and `gfxoff`; its eager Python imports require an
extracted-prefix dependency-closure test. Optional calibration tooling
is an explicit packaging choice. Host-specific
`BIGCHERRY_ENVIRONMENT`, `BC_WORK_ROOT`, `BC_MODEL_ROOT`, models,
route heat, SDK and hardware stay external. **`bc-serve` stays exclusively
in the engine bundle**, alongside `bin/radiance`, the MoE helper,
plugins, catalogs and runtime libraries.

A future pair would require two verified tag/asset identities, per-file
digests, matching ROCm SDK, engine plugin ABI, matching serving-manifest
contract and topology version, compatible layout/architecture inputs
and a fully verified two-prefix install before publication. Do not
confuse the engine *archive* schema
`bigcherry-radiance-runtime-v1` with the native serving manifest's
`apiVersion=bigcherry/v2` and `contract_version=2`; the generated
serving contract fingerprint and actual plugin ABI must be checked.
The proposed `install.json` **v2** records both tags/tag commits,
archive and manifest digests, prefixes, engine build and `bc-serve`
digests, build-provenance trees, SDK version, compatibility contract,
selected launcher and a canonical pair ID. The engine tag's
`config_tree` is its **build input**, not the independent platform
config tree. Tags/asset provenance must be authenticated; a checksum
alone does not establish publisher identity.

Staging and active-pair selection would be atomic. A failed upgrade
must leave the old pair active, and rollback re-verifies the retained
pair. Existing v1 receipts with `worktree`/`build_dir` are not silently
rewritten; migration must be explicit. The only previously published
unscoped engine archive (`bc-radiance-1170.0.0`) retains its existing
pinned-digest **v1 exception** and is **not** eligible as a new scoped
v2 pair. No new `--engine-tag`/`--platform-tag` syntax or `latest`
pair-selection policy is implemented; both remain design decisions.
The first planned implementation is a hermetic platform
archive pack/verify plus extracted-prefix tests, **without changing
the current installer**.

## Bump gates per engine

`pin-bump` runs the gates the engine declares through its adapter: llama.cpp the ggml-cuda contract compile and the ggml
audit; radiance the component gate (build the engine and its components, every plugin's `rad_plugin_abi_version` equals
the source's `RAD_ABI_VERSION`, component offline tests) and its own audit. Details: [PIN_BUMP.md](PIN_BUMP.md#per-engine-bump-gates).

## Smoke order for a bump or any change

1. **CPU smoke (RAM-only, no GPU).** `python -m bigcherry --engine <name> smoke-cpu --build <dir>` serves a tiny model on
   the CPU with every GPU hidden and checks three greedy completions twice. `pin-bump` runs it as the `cpu-smoke` phase
   on the build its `lane-build` phase made (`--build`/`--platform`; `--cpu-smoke-build <dir>` overrides, only `--skip-cpu-smoke` omits it, both recorded in the run state). It is also the quickest local
   check after any patch or component change. Models and digests are declared in `[smoke.cpu]` of each `engine.toml`.
2. **Hardware acceptance against stored baselines (MEN17).** `python -m bigcherry --engine <name> accept --output <dir>`
   runs the engine's declared `[[acceptance]]` configs (speed A/B, KLD) through the server comparison and compares the
   numbers with the baselines in `docs/reference/engines/<name>/baselines/`, exiting non-zero on any regression. It needs
   the GPUs, so it is not part of the default bump: `pin-bump --accept` runs it after the CPU smoke and writes the result to
   the release record. Run it on the lab host with `bigcherry remote run`.
3. GPU smoke and the hardware evidence recorded by `pin-release` come after it and only when it passes.

## Versions

A release version is `<major>.<minor>.<patch>`. The major comes from the pin, by the engine's `[release]
version-source` in `engines/<engine>/engine.toml`; the first release of a pin is `<major>.0.0`, and while the pin stays
`--bump minor` (features) or `--bump patch` (fixes) raise the rest. A pin bump that changes the major starts over at
`<new major>.0.0`.

| Line | version-source | Major | Readable alias (added by the release workflow) |
|---|---|---|---|
| llama.cpp | `build-tag` | the upstream build number: b11474 -> 11474 | `bc-llamacpp-b<build>-r<N>` |
| radiance | `pinned-version` | the upstream version's digits, from `pinned-version` in `engines/radiance/recipes.toml`: 1.3.0 -> 130 (an upstream 1.4.0 starts at 140.0.0) | `bc-radiance-<upstream version>-r<N>`, e.g. `bc-radiance-1.3.0-r1` |
| platform | - | plain semantic version from conventional commits | none |

`N` counts the releases at that major from 0. The radiance pin is a commit, so its commands take the revision (at least 7
hex digits of the recipes' `pinned`, e.g. `89cee7ce4144`) where llama.cpp's take the build tag; the release record stays keyed by
the 12-digit revision (`releases/89cee7ce4144.json`) and the `pin-release` note in it states both the revision and the
upstream version. `pinned-version` is a declared recipes field, updated with `pinned` by the bump.

Decision (owner, 2026-10-17): "upstream as major" for radiance, as above. The manifest starts the radiance package at
`130.0.0`; its first release is stated with `release-as` by `pin-release`, as for llama.cpp.

## What a radiance release asset is called

A radiance runtime bundle is BigCherry's build of the radiance engine (the pinned upstream release with BigCherry's patches and
components), not the upstream release, so its assets are named `bigcherryrad-<plan id>` (`.tar.gz`, `.tar.gz.sha256`,
`.receipt.json`) and its `bundle.json` says `product: bigcherryrad` and what it was built from (the upstream project, URL and
pinned tag). The files inside keep the engine's own names (`bin/radiance`, `radiance_home`), because scripts and the engine find
each other by them. Bundles a lab `pin-release` packed before this name existed are `radiance-<plan id>.tar.gz` and still install.

### How a release gets its bundle

When release-please publishes a `bc-radiance-*` release, `.github/workflows/radiance-release-bundle.yml` runs (it ignores every
other release). It decides first whether the runtime is already bundled: the runtime key is the git trees of `engines/radiance`
and `config`, the trees of the build tooling (`BUILD_INPUTS` in `tools/bigcherry/release/ci_bundle.py`) and the ROCm version of
the build image. If an earlier release has a receipt with the same key, that archive is attached again and the receipt names the
new tag as `release_tag` and the earlier one as `reused_from`; otherwise the tag is built in the ROCm 7.2.4 image, the card-free
component tests run, the lane build is packed, the archive is restored into a scratch directory to prove it installs, and the
archive, its `.sha256` and its receipt are attached (the build starts from main's compiler cache, so only what the tag changed
recompiles; most releases only change documentation, so most are reuses).

`bigcherry --engine radiance release-install <tag|latest>` installs it: checksum, per-file digests, source trees against the
tag, the ROCm version against the host SDK. The bundle includes `bigcherry-build-metadata-radiance.json`, which the launcher's
SDK check and the campaign tools read. The first CI bundle (bc-radiance-1170.11.0) was installed on the lab host and brought up
`flashnext-4gpu` (R9700 engine, two RX 7900 XTX and the RX 6900 XT tail on libr2) before the trigger was turned on.

A run that fails leaves the release without assets and the next `release-install latest` skips it. Fix the cause and run the
workflow again from the Actions tab with the tag (`workflow_dispatch`). To replace a bundle that was attached in error, delete its
three assets from the release first: the workflow refuses a release that already has one. Retention (owner decision): the bundles of the newest 10
distinct runtimes are kept (a release that reuses a bundle counts with the release it reuses), the newest radiance release's
always, and any release whose notes have a line `keep-bundle`; the assets of older releases are deleted. The last job of the
workflow prints that plan on every run and deletes only when the repository variable `RADIANCE_BUNDLE_PRUNE` is `true`, so
turning deletion on is an explicit decision. A bundle is about 240 MB compressed.
