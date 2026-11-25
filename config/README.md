# Shared platform configuration

This directory contains **platform-owned** configuration only, with clearly
identified migration exceptions. It is not a default home for engine-specific
source, experiment provenance or historical run outputs.

Shared contracts retained here:
- `environment.example.toml` — a template for host-local SDK and work-root
  settings, never host secrets or installed artifacts.
- `release.toml` — multi-engine release lines and the shared release ledger.
- `jobs/` and `slurm/` — shared job execution and scheduler templates.
- `forbidden-terms-allow.txt` — centrally reviewed hygiene allowlist.

**Declared exceptions pending a separate compatible migration:**
`recipes.toml` is the llama.cpp campaign/pin recipe currently named by
`engines/llamacpp/engine.toml`, and
`experiment-contracts.toml`, `experiment-contract-legacy.toml`, and
`models.toml` currently describe llama.cpp patch qualification.
These are **not** examples for new engine-specific configuration. They must
move together with owner-aware path resolution, release/build provenance,
frozen waiver handling, CI and operator acceptance. See
`docs/reviews/repository-layout-2026-11-20/README.md` for the scoped plan.

Engine-specific registries already live with their engine:
`engines/llamacpp/external-sources.toml` and the non-runtime
`engines/radiance/reference/serving-settings-inventory.toml`.
Radiance's real serving configs live under `engines/radiance/configs/`,
and its authoritative native parser/schema under
`engines/radiance/components/serving/`.

Do not create a root `host/`, `sql/`, `dispositions/`, `out/` or root
`patches/` tree as a maintained source location. Direct reusable platform
code and schema to its consuming component, and archive experimental evidence
under a named, reviewed experiment or compact evidence bundle.
