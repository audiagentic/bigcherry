-- BigCherry dispatch DB schema 6 -> 7 (HI121).
--
-- Adds backend-scoped build_capability: a durable record of what semantic
-- distinctions a build's producer code actually knew how to evaluate,
-- source-owned (src/ggml/src/ggml-cuda/hip-autotune-types.h's
-- GGML_HIP_PRODUCER_CAPABILITIES_LO/HI) and never inferred from
-- signature_schema, source_revision, or commit date. Deliberately does NOT
-- backfill build_capability rows for existing builds -- absence of a row
-- means capability-unknown, not an inferred default, and no historical
-- measurement was ever attested this way. That is intentional, not an
-- oversight: HI121's whole point is a producer's capability claim must come
-- from verified provenance, and no such verification exists for builds
-- recorded before this migration.
--
-- Also fixes a real pre-existing bug found while designing this migration:
-- winner_dispatch_idx and vk_winner_dispatch_idx were both wrongly GLOBAL
-- unique indexes, even though each table's own row-level UNIQUE constraint
-- is already build-scoped -- the global index prevented two different
-- builds from ever both retaining a winner for the same portable dispatch,
-- which HI121's multi-generation measurement reuse needs to allow.
--
-- This migration is applied via Python's sqlite3.Connection.executescript()
-- (see tools/tests/campaign/test_db_migration.py for the established real
-- invocation pattern for this project's migrations) -- executescript()
-- stops at the first statement that raises, so the schema-version guard
-- below (a CHECK-constraint violation on a temp table) genuinely aborts the
-- whole script rather than silently continuing on a version mismatch.

PRAGMA foreign_keys = ON;

BEGIN;

-- Fail closed unless this database is currently EXACTLY schema 6. A CHECK
-- violation here is a real SQLite error that halts the rest of this script.
CREATE TEMP TABLE _hi121_schema_guard (
    version TEXT NOT NULL CHECK (version = '6')
);

INSERT INTO _hi121_schema_guard(version)
VALUES ((SELECT value FROM schema_meta WHERE key = 'schema_version'));

DROP TABLE _hi121_schema_guard;

CREATE TABLE build_capability (
    build_id              INTEGER NOT NULL REFERENCES build(build_id),
    backend               TEXT    NOT NULL,
    producer_capabilities BLOB    NOT NULL,
    PRIMARY KEY (build_id, backend),
    CHECK (length(producer_capabilities) = 16)
);

DROP INDEX winner_dispatch_idx;

CREATE INDEX winner_dispatch_idx
    ON winner(dispatch_digest, objective);

DROP INDEX vk_winner_dispatch_idx;

CREATE INDEX vk_winner_dispatch_idx
    ON vk_winner(dispatch_digest, objective);

UPDATE schema_meta SET value = '7' WHERE key = 'schema_version';

COMMIT;

PRAGMA foreign_key_check;
