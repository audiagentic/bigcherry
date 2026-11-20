# llama.cpp engine ownership

`engine.toml` is the path and upstream declaration for this engine.

- `patches/` and `overlay/` are maintained engine source. Their content
  is protected by patch digests, recipe composition and CI gates.
- `dispositions/` is the Git-tracked narrow patch-failure coverage store;
  see its README for expiration and review, not a permanent waiver.
- `external-sources.toml` is the engine-owned registry of upstream
  forks/PRs, exact snapshots and tracked patch provenance. Use
  `bigcherry --engine llamacpp sources status` for an offline view and
  `sources check` for explicit read-only remote drift checks. The current
  path is declared by `engine.toml`; generic platform readers must derive
  it rather than retain the former `config/external-sources.toml` literal.
- The build/campaign `recipes.toml` currently resides at
  `config/recipes.toml` for compatibility with existing release/build
  receipts and CI. It is an acknowledged migration exception, not the
  desired ownership boundary; do not duplicate it or change its authority
  without an exact staged migration and full parity/recovery gates.

Shared `tools/bigcherry/` provides the engine-agnostic API and producer
contracts; engine-specific patch code stays under this directory. Historical
measurement evidence belongs under `docs/evidence/` with provenance.
