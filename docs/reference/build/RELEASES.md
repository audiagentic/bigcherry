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

## Installing a release

`release-install <tag|latest>` (Radiance) installs a release's runtime bundle into a clean detached worktree at the
tag; see the [Radiance runbook](../engines/radiance/RUNBOOK.md) for paths and launch.

- A `runtime_assets[<tag>]` entry in a pin record committed at the tag is the source of truth and takes precedence
  (receipt `source: "record"`).
- A tag with no entry (a release made by CI) installs from its release's own `radiance-<plan>.tar.gz`, `.sha256` and
  `radiance-<plan>.receipt.json` (receipt `source: "ci"`). The receipt is the asset entry plus `release_tag`,
  `runtime_key` and an optional `reused_from`. It installs only when the archive matches the `.sha256` asset and the
  receipt; the receipt's `release_tag` is the tag; `runtime_key`'s `engine_tree`/`config_tree` equal the tag's trees
  (git objects at the tag) and the bundle's; every extracted file matches the archive's `bundle.json`; and
  `rocm_version` equals the host SDK. `bundle.json` may name another tag only when `reused_from` names that tag and
  its trees equal this tag's. Any failure refuses the install.
- `latest` is the numerically newest local `bc-radiance-*` tag with a record entry or a release carrying exactly one
  archive with its `.sha256` and receipt. A bundle that then fails verification refuses; it does not fall back to an
  older tag.

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
