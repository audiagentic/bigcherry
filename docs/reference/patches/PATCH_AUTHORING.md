# Patch authoring

Choose a flat module for a simple patch. Choose a package when the patch owns
validation configuration or helper files.

Every package must contain `patch.toml` and `patch.py`; `validation.toml` and
`validation/` are optional. Keep patch IDs, implementation files, and metadata
consistent. Do not put Experiment Contract fields (hypothesis, workload,
controls, boundaries, thresholds, or hardware claims) in adapter configuration.

Before deleting a flat module during migration, apply both representations to
the same immutable base and require identical source-tree hashes.
