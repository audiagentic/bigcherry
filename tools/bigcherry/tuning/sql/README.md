# Shared tuning dispatch database schema

**Owner:** `bigcherry.tuning` (cross-engine tuning/inventory platform). These files are **maintained source**, not runtime output or an engine-specific vendor patch.

- `dispatch-db.sql` is the canonical bootstrap schema for tuning and record databases, including HIP and Vulkan families.
- `migrations/*.sql` are immutable historical schema transitions for deliberate, tested upgrades and examination of older databases.
- `tools/bigcherry/core/paths.py:SQL` is the single path resolver used by the production CLI, lifecycle, workflow and tests.
- Runtime databases (`*.db`, `*.sqlite`, `*.sqlite3`, `dispatch-db/`) belong under machine-local work or artifacts, never here.

**Change contract:** change the bootstrap DDL and schema metadata together; add or reconcile an explicit migration; verify fixtures and old-to-current migration tests; prove the HIP/Vulkan ABI declarations still agree. Refuse unknown schema versions. Do not add scratch SQL dumps, direct database snapshots or ad hoc experiments here. Engine-specific runtime code stays within that engine; this SQL is the intentionally shared exchange schema.

Historical evidence and recovered DDL are retained separately under `docs/recovery/`.
