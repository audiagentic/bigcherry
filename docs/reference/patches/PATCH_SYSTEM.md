# BigCherry patch system

Every production patch is a package directory named after its patch ID.
Packages contain `patch.toml` for identity/metadata and `patch.py` for the
implementation. Optional `validation.toml` and `validation/` content own
patch-specific validation; implementation and validation identity are digest
bound. The registry still understands legacy flat modules for compatibility
fixtures, but new production patches must not use them.

Use `bigcherry patch-lint` for non-mutating metadata checks, `bigcherry check`
for deterministic local CI, and `bigcherry patch-validate` to inspect existing
validation evidence. Hardware campaigns are explicit and are never launched
by `bigcherry check`.
