# Radiance bump-readiness milestones (2026-11-18)

Owner scope: protect the BigCherry Python-removal enhancements for the separate RUM01–RUM10 upstream migration. This lane does **not** bump the vendor tree or implement RUM's ports. No GPU tests.

## Milestone 1 — Python-free release provenance

- The single RSP01 Python-free asset manifest and asset record now capture `built_from.pinned`, `pinned_version`, and a `components` map of **all six** named component `built-against` revisions, read from `engine.toml` / `recipes.toml` / `component.toml` at pack time.
- Fail closed when a component declaration is missing or diverges from the Radiance pin; the asset and manifest must agree on build identity. A pin move changes manifest metadata with no hard-coded version edits.
- Validation: `tools/tests/release/test_bu01_runtime_assets.py`: 22 passed, 16 subtests (card-free), including target pin change and stale/missing component regressions.
- The release schema/metadata still reflects the current v1.17.0 pin; RUM03 owns full 2.x plugin-ABI migration. Standalone hardware acceptance remains with RSP01/RUM09.

## Pending milestones

1. Source/header/flag/environment/ABI coupling inventory and fail-closed core gate, mapped to RUM02–RUM10.
2. Verify the existing typed v2 `radiance_option` / `env_setting` registry as the one translation seam consumed by C++ and rad0009; keep C++ changes small during concurrent RSC02 step 6.
3. Read-only newest-tag pin triage plus scratch extraction / exact rewrite checks; annotate `BUMP_COUPLING.md`.
4. Bump skill and build-guide procedure update; request plan notes through Planning MCP when available.
5. Full Python suites and associated static validation, independent PR merge / CI confirmation.
