# llama.cpp

BigCherry keeps [llama.cpp](https://github.com/ggml-org/llama.cpp) close to
upstream while developing HIP performance work and targeted fixes.

## What changes

The HIP measured-dispatch path records matrix-operation signatures seen by a
real workload, evaluates candidate kernels against a pinned build, and
publishes selected winners for replay. Serving replay uses a compact cache
and falls back to native dispatch when there is no matching winner.

Changes to upstream-owned files are **patch packages**. New files belong in
the engine's **overlay**, with paths matching upstream. The two can be
reapplied after an upstream pin bump; they are not a standing fork.

## Browse the changes

The [patch catalog](PATCHES.md) lists every package, its recorded state and
a link to its own page. That package page owns the mechanism and evidence;
the catalog does not make independent performance claims.

For the underlying systems, see the
[patch system](../../patches/PATCH_SYSTEM.md),
[dispatch and candidate architecture](../../architecture/MMQ_CATALOG.md),
[patch validation](../../testing/PATCH_VALIDATION.md) and
[build and pinning guide](../../build/PIN_BUMP.md).

## In the repository

- `engines/llamacpp/patches/<id>/` — patch metadata, implementation,
  README, rationale and evidence.
- `engines/llamacpp/overlay/` — files BigCherry adds to upstream.
- `engines/llamacpp/recipes.toml` — source and patch-set selections.

For commands, start at [Getting started](../../../../GETTING_STARTED.md).
