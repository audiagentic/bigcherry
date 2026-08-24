# BigCherry patch system

The patch system supports both root-level flat patch modules and packaged
patches. Flat patches remain first-class for small, self-contained changes.
Packaged patches use `patch.toml` for identity and `validation.toml` for
validator configuration; implementation and validation identity are digest
bound.

Use `bigcherry patch-lint` for non-mutating metadata checks, `bigcherry check`
for deterministic local CI, and `bigcherry patch-validate` to inspect existing
validation evidence. Hardware campaigns are explicit and are never launched
by `bigcherry check`.
