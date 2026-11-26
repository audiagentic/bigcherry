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
   binary, matching `radiance_home/bin/bc-serve` and `bc-release`, plugins, kernel
   catalogs, helpers, runtime shared libraries and committed RSC12 configs. Diagnostic `rad-*` binaries and
   `r4d_selftest` stay out. Only explicitly recorded published legacy
   unscoped assets retain their historical verification allowance.

## Installing a release

Radiance releases now use a single Python-free **engine-runtime** asset and native `bc-release install/verify/rollback`. The platform CLI does **not** offer the retired `release-install` command and does not create detached source worktrees for deployment. Native `bc-serve` in the verified current prefix rechecks the tag-bound installation before starting; no source-checkout launcher or Python runtime is required.

Obtain an exact release tag's `bigcherryrad-PLAN.tar.gz`, `.sha256` and `.receipt.json` from the same trusted publishing channel. The receipt's tag and cryptographic binding must be verified, not inferred from a mutable `latest` pointer.

**Native install and CI release workflow (RSP01 phase-2 slices):**
`radiance_home/bin/bc-release` is an independently linked C++20 release
manager (zlib, vendored header-only JSON, existing SHA-256). It has no
Python, Git, shell, GPU or checkout dependency at execution. Its
`--record` input must be the receipt retrieved from the **selected exact
`bc-radiance-X.Y.Z` release**, not a mutable latest index or a record
invented by the local install host. The native tool verifies this receipt's
exact tag and product/scope, required archive filename, SHA-256, matching
manifest/file digests and engine/serving identities before accepting any
runtime file. A receipt whose authenticity was not established by its
tagged publishing channel cannot itself prove provenance. Online
release-record acquisition belongs to the workflow cutover, not a
fallback to repository Python.

From the root of an extracted, verified engine asset:

```sh
radiance_home/bin/bc-release install --tag bc-radiance-X.Y.Z \
  --record /path/to/tag-bound-release.receipt.json \
  --archive /path/to/bigcherryrad-PLAN.tar.gz \
  --root /path/to/installed-radiance --sdk /path/to/rocm-sdk
radiance_home/bin/bc-release verify --root /path/to/installed-radiance --sdk /path/to/rocm-sdk
radiance_home/bin/bc-release verify --root /path/to/installed-radiance --sdk /path/to/rocm-sdk
radiance_home/bin/bc-release rollback --root /path/to/installed-radiance --sdk /path/to/rocm-sdk
```

The isolated install root contains immutable `releases/<tag>/` prefixes,
their native `install.json` receipts, and an atomically replaced
`index.json` with exactly **current** and **previous** entries. Any
corrupt current or previous prefix prevents switching or rollback.
Native `install` refuses an older semantic version (or an equal numeric version with a distinct exact tag) before extraction unless the operator explicitly passes `--allow-downgrade true`. The installer compares decimal version components numerically, not lexically. Authorized installs still retain the displaced verified current as `previous`; `rollback` is an independently explicit verified switch that needs no downgrade flag.

Interrupted `.staging-*` trees cannot become a current release or be
implicitly reused. The native error families distinguish invalid input
(exit 2), digest failure (3), identity mismatch (4), unsafe archive (5),
corrupt installation (6), conflict (7), I/O (8), and SDK mismatch (9).
Rollback only changes the verified index; it **does not start** an engine,
assume hardware health or bypass `bc-serve` live guards. RSC02 step 6 and
the RSC04 acceptance remain prerequisite to the production cutover.

**Installed-tag startup:** run `releases/<tag>/radiance_home/bin/bc-serve
serve CONFIG --run-dir DIR` (or `start CONFIG --run-dir DIR`) from the
**current** verified release. The running native executable derives its
installation prefix from its own canonical path, verifies the native
`bc-release` current/previous index, tag-bound receipts, matched SDK and
**every installed payload digest** before it plans or spawns children, and
rejects absent metadata or a retained but non-current tag. `start` and its
detached `serve` owner independently recheck the same identity. If neither
`--configs` nor `BIGCHERRY_CONFIGS` is supplied, named configs come from
that verified prefix's `configs/`; the engine binary and runtime home also
default to this exact prefix. Explicit config arguments and environment
retain their precedence, but cannot bypass release verification or substitute
a different build's engine/home. A development binary not inside a verified
release requires explicit configs. The native verifier is linked directly
into `bc-serve`, not reimplemented or launched through Python/shell.
For an installed production Flash-Next start, only the model location and
run directory need host paths (plus any host-specific site/calibration inputs):

```sh
PREFIX=/path/to/installed-radiance/releases/bc-radiance-X.Y.Z
"$PREFIX/radiance_home/bin/bc-serve" serve flashnext-4gpu \
  --model-root /path/to/models --run-dir /path/to/runs/flashnext-production \
  --power-profile stock
```

The released `configs/flashnext-4gpu.yaml` (like each shipped config) selects the reviewed
`@bundle/heat/flashnext-mixed.hot-first` seed under the same verified prefix.
Both this seed and `perf51.rank0.hot-first` ship as immutable, manifest-digested
`heat/` files copied from their tagged repository sources. No repository
pathname or `--set handoff.heat` is needed. With helpers, the session owner
defaults recording to `<run-dir>/heat.live` at a 1-second interval, and passes
the same effective path through planning and the manifest writer. A site,
profile or `--set` may explicitly override either path; `handoff.histogram
= "off"` explicitly disables recording. These defaults do not replace
hardware acceptance, the installed-verification gate or a separately selected
live-policy calibration input.

The card-free install test also runs the real detached session owner against
the fake engine **installed by `bc-release`**, while actual GPU acceptance
remains gated on leased S15.

The declared native `bc_release_install_test` CTest builds freshly versioned
card-free archives and exercises install, verify, switch, rollback, tampering
and other refusal cases. It also installs a pinned PAX/gzip archive generated
by the real production `runtime_assets.pack()` writer, rather than merely
testing its own synthetic tar format. No GPU or chmod/permission inference
is required. Release CI additionally checks a newly packed **or reused**
archive's exact member file set, every digest and mode, the two RSC12
configs and both reviewed heat seeds against the selected tag, the tag's engine/config/build-input trees,
the payload contract and absence of Python/shell/interpreter dependencies
before attaching its three files (`.tar.gz`, `.sha256`, `.receipt.json`).
Python is needed only by the CI-side builder and publisher, never in the
engine asset or by the native installer.

**Engine payload invariant:** The
runtime file manifest includes the two RSC12 TOMLs from the committed engine
source, not from mutable build/site directories. A pack/restore gate rejects
missing or unreviewed configs, script files, Python caches, Python/shell
shebangs and interpreter references in human-readable runtime metadata.
The extracted prefix has `configs/` without a source checkout, platform
package or launcher. This is an **asset invariant**; hardware dense/helper acceptance is still separate from offline installation verification. An archive from an older tag without
these files is not a valid new phase-2 engine asset.

The scoped release asset contains only engine **binaries and data**: `bin/radiance`, matching engine-built `radiance_home/bin/bc-serve`, MoE helper, kernel/architecture/quantizer plugins, required runtime libraries and metadata, plus `configs/flashnext.toml` and `configs/qwen27b.toml` at the archive root (copied from the tagged `engines/radiance/configs/` source) and the reviewed, digest-bound `heat/` seeds. `engines/radiance/archs.toml` is an **engine build input compiled into `bc-serve`** (`components/serving/CMakeLists.txt`); the current `engines/radiance/layouts/*.toml` become RSC12 configs in the bundle, not a platform-supplied layout overlay. Exclude Python, the Python supervisor, shell launcher, build/patch/campaign tools, `rad-*` diagnostics, `r4d_selftest`, tests and mutable/private host files. The independent `bc-platform-*` release is never selected or installed to run this asset.

**Architecture-plugin retention (RSP01 F2c review, 2026-11-18).** The upstream loader scans and stages every `architectures/*.so` and does not apply the `--kernels` filter to architecture or quantizer plugins. The shipped configs identify external `.rad` model files, not their embedded architecture-plugin and quantization IDs. Without checking all pinned model headers and their supported fallback IDs, no smaller architecture allowlist is proven to preserve loading. Retain the complete architecture plugin directory rather than infer a family from a model filename. This is a deliberate compatibility reservation, **not** a claim that the bundle is model-minimal; shrinking it requires a reviewed model-to-plugin inventory and a coverage test for every shipped config. `kernels.rkb` and `kernels_moe.rkb` are benchmark-only fixtures and are excluded from the runtime manifest and native installer.

Native `bc-serve start CONFIG` / `serve CONFIG` / `stop` / `status` (RSC02 and RSC12) must own config resolution, the memory plan, ordered helpers, host checks, power/gfxoff, calibration and heat placement with no fallback to Python. The native session owner, helpers and guards have landed through RSC02 step 5, but final step-6 native-mode parity and RSC04 hardware acceptance are not complete; the obsolete shell/Python start commands have been removed from the development tree. Host ROCm SDK, hardware, models, site configuration and selected heat/policy/calibration inputs remain external.

Rollback retains the **current and previous** verified engine release, with the previous reverified on rollback. **Existing installs are never converted** or their v1 receipts rewritten; a fresh reinstall from a tag replaces other older installs under the two-release retention policy. The one previously published unscoped `bc-radiance-1170.0.0` archive keeps exactly its phase-1 SHA-pinned `PUBLISHED_LEGACY_ASSETS` exception, not eligibility for the new standalone format. Unsafe or partial upgrades never displace the active verified release. The native installer, current/previous retention and atomic rollback are
implemented. Production `bc-serve serve CONFIG` from a newly installed tag
still awaits RSC02 step-6 helper/health parity and the leased hardware gate.
The legacy Python installer and shell launcher have been deleted from the source-deletion branch; neither is bundled with the engine. Do not merge before the S15 hardware gate.


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
| radiance | `pinned-version` | From `pinned-version` in `engines/radiance/recipes.toml`. Published 1.x retains concatenated digits (1.17.0 → 1170). From 2.x the same four digits continue: upstream major × 1000 + minor × 10 + patch (2.2.0 → 2020, 2.10.1 → 2101); a minor of 100 or more, or a patch of 10 or more, is refused because it would collide. | `bc-radiance-<upstream version>-r<N>`, e.g. `bc-radiance-1.3.0-r1` |
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
archive, its `.sha256` and its receipt are attached only after the mandatory
Python-free, interpreter/shebang, exact-tag content and digest verification (the build starts from main's compiler cache, so only what the tag changed
recompiles; most releases only change documentation, so most are reuses).

The native `bc-release install --tag TAG --record RECEIPT --archive ARCHIVE --root ROOT --sdk SDK` validates the selected exact tag, checks every file, SDK and installed index, and retains current/previous prefixes. The first CI bundle (`bc-radiance-1170.11.0`) was validated on a lab host; that earlier historical run is not evidence for the S16 cutover.

A run that fails leaves the release without assets and there is no installable asset for that tag. Fix the cause and run the
workflow again from the Actions tab with the tag (`workflow_dispatch`). To replace a bundle that was attached in error, delete its
three assets from the release first: the workflow refuses a release that already has one. Retention (owner decision): the bundles of the newest 10
distinct runtimes are kept (a release that reuses a bundle counts with the release it reuses), the newest radiance release's
always, and any release whose notes have a line `keep-bundle`; the assets of older releases are deleted. The last job of the
workflow prints that plan on every run and deletes only when the repository variable `RADIANCE_BUNDLE_PRUNE` is `true`, so
turning deletion on is an explicit decision. A bundle is about 240 MB compressed.
