# Patch and component documentation

The package owns its documentation. An engine overview introduces the work;
a generated catalog links to each package; the package page explains the
actual change and its evidence. Do not copy technical detail or performance
tables into summary pages.

## Location and required sections

- **llama.cpp patch:** `engines/llamacpp/patches/<id>/README.md`.
- **Radiance patch:** `engines/radiance/patches/<id>/SUMMARY.md` (or an existing
  `README.md`; do not create duplicate pages).
- **Radiance component:** `engines/radiance/components/<name>/README.md`.

Each page uses these exact level-two headings, in this order:

```markdown
## Summary

One or two plain sentences describing what changed and where.

## Outcome

The recorded state, qualification scope and evidence or current blocker.
If the result is unknown, say so. Never turn metadata into a measured claim.

## Detail

Mechanism, constraints, testing, evidence and further links.
Other subsections may follow.
```

Write for the next engineer: identify the behavior, the boundary of the
change, whether it is enabled, and what is actually proven. Use specific
names, measured quantities where supported, and direct links. Avoid
marketing language, repeated introductions and speculative speedup claims.

## Lifecycle sources of truth

A patch's `patch.toml` `state` is authoritative for its lifecycle, with
`SUMMARY.md` retaining its status and evidence record. States are
`untested`, `validated`, `rejected`, `evaluated` and `superseded`.
`validated` is a **package state**, not permission to claim universal
hardware coverage; read the evidence scope and selection in the engine recipe.

A component has its own `component.toml` settings. Different settings can
be production, experimental, evaluated, diagnostic or tuning, and some
components record a validated setting. Do not invent a single production
state for a component. Describe current settings accurately and link their
evidence; the component catalog therefore says "See settings".

For a legacy package, preserve the original analysis under `## Detail`.
If its state or measurements are historical, mark the limitation instead
of silently rewriting the recorded result.

## Generated catalogs

`tools/docs/generate_engine_enhancement_summaries.py` reads the package
docs and metadata and writes only the marked sections in:

- [llama.cpp patch catalog](engines/llamacpp/PATCHES.md)
- [Radiance patches and components](engines/radiance/ENHANCEMENTS.md)

Run `python tools/docs/generate_engine_enhancement_summaries.py` after
changing package docs. Run the same command with `--check` to verify
headings, metadata identifiers and catalog freshness without writing.
Commit the generated catalog alongside the owning documentation.

GitHub does not inline other Markdown files. Catalog generation is explicit;
engine overview pages remain short and link to the generated catalogs.
