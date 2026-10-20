# bigcherry

Measured-dispatch autotuner and release platform for two inference engines:
**llama.cpp** (primary, patch-overlay model) and **radiance** (second engine,
multi-GPU AMD serving). bigcherry keeps its own work release-tolerant rather
than forking either engine.

New to this repo? Start with `GETTING_STARTED.md` for build/test commands and
operational gotchas.

## How the pieces fit

- **`tools/bigcherry/`** — the `bigcherry` Python CLI. Every engine (build,
  patch, test, validate, promote, release) goes through its commands:
  `python -m bigcherry --engine <llamacpp|radiance> <command>`.
- **`engines/llamacpp/`** — llama.cpp kept current via a two-bucket overlay
  instead of a long-lived fork:
  - `src/` — whole files new to the tree, mirroring the upstream layout
    exactly, so applying it is a copy and the eventual upstream PR diff falls
    straight out of it.
  - `patches/<id>/` — small, anchored edits to files upstream already owns.
    Each is a self-contained package (`patch.toml`, `patch.py`, `SUMMARY.md`,
    `README.md`); see
    [`docs/reference/patches/PATCH_SYSTEM.md`](docs/reference/patches/PATCH_SYSTEM.md)
    and
    [`docs/reference/patches/PATCH_AUTHORING.md`](docs/reference/patches/PATCH_AUTHORING.md)
    for the full package layout and lifecycle.
- **`engines/radiance/`** — the second serving engine: engine patches plus
  `components/` (BigCherry source built against the installed engine, not a
  patch to it). See
  [`docs/reference/engines/radiance/README.md`](docs/reference/engines/radiance/README.md).
- **`docs/reference/`** — maintained cross-cutting guidance: patch system,
  per-engine overviews and per-model pages, the per-change doc shape
  ([`docs/reference/DOC_STRUCTURE.md`](docs/reference/DOC_STRUCTURE.md)), and
  architecture detail (e.g.
  [`docs/reference/architecture/MMQ_CATALOG.md`](docs/reference/architecture/MMQ_CATALOG.md)
  for the MMQ candidate catalog and architecture targeting).
- **`docs/planning/`** — active/completed plan items, design decisions, and
  work history, managed through the `ag-planning` tooling.
- **`docs/releases/`** — human-facing release notes, the release/change
  ledger, and audit/check-in guidance for cutting a release.

## Workflow (llama.cpp)

```bash
python -m bigcherry --engine llamacpp pull   --ref b1234        # fetch/checkout upstream
python -m bigcherry --engine llamacpp audit                     # strict invariant audit
python -m bigcherry apply --source bigcherry  # engines/llamacpp/overlay + anchored patches
python -m bigcherry generate --arch all       # candidate catalog -> artifacts
python -m bigcherry build   --profile replay-full
```

Every stage is idempotent, and every stage after `audit` refuses to run on a
tree that has not passed a strict audit for its current revision. Radiance
follows the same build -> patch -> test -> validate -> promote -> release
workflow through its own engine-aware commands; see
[`docs/reference/engines/radiance/README.md`](docs/reference/engines/radiance/README.md#workflow).
