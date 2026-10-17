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

   Generated from the engine's own events since its previous release tag, the engine's patch registry changes (added,
   promoted, changed, retired, entered or left the production build) and the bump's release record
   (`releases/<pin>.json`). Output: `docs/releases/notes/<tag-prefix><version>.md`. Never edit it by hand.
3. **release-please PR.** The notes commit sets the engine package's `release-as` in `release-please-config.json`;
   after it reaches `main`, release-please opens that package's release PR (changelog, manifest version).
4. **Merge and tag.** Merging the release PR makes the tag of record `<tag-prefix><version>` and the GitHub release.
   The release workflow (`.github/workflows/release-please.yml`) then titles the release and attaches the generated
   notes file.
5. **Pin-release record.**

       python -m bigcherry --engine <name> pin-release <pin> --evidence @<file> [--bump minor|patch] [--dry-run]

   runs stages 2 to 4 as resumable phases (`gate`, `record`, `notes`, `main`, `release`, `unpin`, `sync`) and writes the
   gate output and hardware evidence into the pin's release record. `--dry-run` prints the concrete actions and writes
   nothing. See [PIN_BUMP.md](PIN_BUMP.md) for the bump that precedes it.

## Versions

| Line | Version | Status |
|---|---|---|
| llama.cpp | `<llama build>.<minor>.<patch>`; a pin bump is `<build>.0.0`, later releases on the pin raise minor or patch; workflow adds the alias `bc-llamacpp-b<build>-r<N>` | in use |
| radiance | not decided: radiance's pin is a commit of a project with its own semantic version (1.3.0 at the current pin), not a numbered build, so the llama.cpp scheme does not carry over. The package starts at `0.0.0` in the manifest (nothing released) | owner decision open (MEN13) |
| platform | plain semantic version from conventional commits | in use |

Until radiance's scheme is decided, `pin-release` for radiance passes the package and release-line gates and then stops
with "no version scheme yet"; `release-notes` for radiance resolves its configuration but its version argument must
follow the scheme once chosen.
