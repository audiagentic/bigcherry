# BigCherry repository ownership sweep — 20 November 2026

Read-only classification of tracked and ignored repository material at
`origin/main` commit `821420af597c44e1e8ada8a7d78830be86ab88e6`,
with a narrow ownership relocation in this PR.

## Applied safely

- Moved the llama.cpp-only external fork/provenance registry from
  `config/external-sources.toml` to
  `engines/llamacpp/external-sources.toml`. The engine's `engine.toml`
  declaration is authoritative; `core.paths.EXTERNAL_SOURCES` derives from
  it. Relevant tests, CI references and maintained docs are updated.
- Moved the static Radiance launcher-setting design inventory from
  `config/serving-settings.toml` to
  `engines/radiance/reference/serving-settings-inventory.toml`. There are
  no production readers. It is **not** a serving config or runtime schema,
  so it stays outside the discoverable `engines/radiance/configs/` tree.
- Removed an ignored, abandoned root `patches/` bytecode-cache tree from
  the primary checkout: 138 `.pyc` files, 1,900,673 bytes; no source files,
  symlinks, Git-tracked entries or active process references. This is local
  scratch cleanup, not a tracked Git change.

## Retain: intentional maintained owners

- `releases/` is the canonical versioned compatibility/rebase state for
  platform release tooling, not a random archive. It contains index and
  history which must remain addressable; **do not** move it until versioned
  engine-qualified record identity, migration and a recovery test exist.
- `config/environment.example.toml`, `config/release.toml`,
  `engines/radiance/power-profiles.yaml`, Slurm and job execution configs serve
  shared platform/host orchestration. They are not engine kernel code.
- `tools/env/` owns ROCm environment implementations. Legacy
  `tools/rocm-env.sh` and `.ps1` are intentional source/dot-source
  compatibility wrappers (RA06). Likewise `tools/residency_gates.py`
  and `tools/verify_slice_a.py` preserve tested lab CLI entry points
  (RA12/RA13). Do not delete without consumer and parity checks.
- `tools/lab/` has 24 named study directories, each with a README;
  `experiments/radiance/` records closed work with owner-specific READMEs.
  Neither should be treated as production runtime by default.
- Untracked `vendor/`, `work/`, `worktrees/` and machine-specific
  metadata are operational/checkpoint data. Other agents or live builds
  may own them; they were **not** pruned.

## Remaining ownership migrations — require separate gates

1. **llama.cpp build recipes:** `config/recipes.toml` is engine-specific,
   but release/pin-bump, build identity, tooling and CI still name it
   explicitly. A search found approximately 110 references in 74 tracked
   files (including historical documentation); at least 16 active tests
   and five CI workflow locations require updates. Migrate to
   `engines/llamacpp/recipes.toml` only as an atomic engine-declaration,
   source/build/receipt and release-compatibility change with exact
   recipe-composition and ABI parity tests. No silent fallback.
2. **llama.cpp experiment contract/model data:** `config/experiment-contracts.toml`,
   `config/experiment-contract-legacy.toml`, `config/models.toml`
   currently describe llama.cpp patch qualification, while their readers
   are a shared experiment platform and include hardcoded defaults. Introduce
   an explicit per-engine registry locator and cross-engine schema contract
   before relocating all three together. Preserve frozen legacy waivers.
3. **Recovered result volume:** `docs/evidence/2026-11-17-work-root-results/`
   contains about 1,774 files / 50,956,518 bytes, including large
   manifests, JSON and logs. Its README documents recovery after the
   original work-root files were deleted. The baseline lacked a bundle
   checksum index; this audit adds a privacy-preserving
   `integrity-digests.tsv` (SHA-256 of relative path and file bytes)
   with an explicitly **noncanonical** `integrity-index.json` for the
   1,775 retained files (including the integrity note), verified
   against current bytes. Historical filenames contain machine names,
   so a plain-text path listing would violate repository hygiene.
   This does **not** reconstruct run provenance or hardware verdicts.
   Before reducing tracked volume, inventory references and migrate
   consumers; never discard the sole copy of successful or failed test
   evidence merely to save space.
4. **Release records:** some `releases/*.json` objects are legacy
   non-engine-qualified records. The directory remains a platform
   read/write contract; partition by engine only with explicit versioned
   identity and recovery tests, not a mass Git rename.

## Continuing standard

Every new tracked source directory must have an explicit owner and
lifecycle. Engine-specific configuration and provenance live with the
engine unless the data is deliberately cross-engine; platform-owned
schemas live by their consumer. Inert historical work goes to a cataloged
experiment or immutable evidence bundle, never a runtime config scan.
Ignored caches and runs stay outside Git; remove them only when no active
process or authoritative source depends on them. Do not rewrite old
release ledgers, archived plans or patch digests to cosmetically repair
historical path strings. For future migrations, test both direct imports
and the native/CI producer-consumer paths before merging.
