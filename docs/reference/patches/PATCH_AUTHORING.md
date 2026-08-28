# Patch authoring

Create a package directory for every patch, including simple patches. The
package directory name must equal the patch ID.

Every package must contain `patch.toml` and `patch.py`; `validation.toml`,
`validation/`, `evidence/`, and patch-local documentation are optional. Keep
patch IDs, implementation files, and metadata consistent. Do not put Experiment Contract fields (hypothesis, workload,
controls, boundaries, thresholds, or hardware claims) in adapter configuration.

Before deleting a legacy flat module during migration, apply both
representations to the same immutable base and require identical source-tree
hashes. The flat representation is retained only for compatibility fixtures.
