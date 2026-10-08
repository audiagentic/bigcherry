# Dispatch database schema

This directory is **source**, not runtime output.

- `dispatch-db.sql` is the canonical bootstrap schema for a new record/tune
  SQLite database.
- `migrations/*.sql` are explicit historical schema transitions that remain
  tracked because migration tests exercise them and older durable databases may
  need to be inspected or upgraded deliberately.
- `docs/recovery/schema9-recovered-ddl.sql` is provenance for a recovered
  historical schema; it is not a runtime migration.
- Runtime databases (`*.db`, `*.sqlite`, `*.sqlite3`, `dispatch-db/`) are
  machine-local and ignored by Git.

A schema change must update the bootstrap schema, add/adjust the explicit
migration path where required, bump the schema metadata consistently, and add
or update migration tests. Do not use this directory for ad-hoc SQL dumps,
scratch queries, or captured production databases.
