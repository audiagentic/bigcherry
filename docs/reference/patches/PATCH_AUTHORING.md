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

Once a patch is ready to enter validation (any tracked-status beyond
`ported-untested`), see
[`docs/reference/testing/PATCH_VALIDATION.md`](../testing/PATCH_VALIDATION.md)
for the required package contents (`README.md`, `validation.toml`,
`evidence/validation.json`), Experiment Contract binding rules, and the
real validation execution architecture.
