-- BigCherry dispatch DB schema 7 -> 8 (HI121 close-out step 6, HI127).
--
-- Adds winner_verification: a per-WINNER (not per-signature, not per-build)
-- attestation that this exact winner row was ingested through the HI125-
-- strengthened path (build capability/descriptor proof passed AND this
-- row's canonical/digest passed real C++ verification). Deliberately does
-- NOT backfill winner_verification rows for existing winners -- absence of
-- a row means not-yet-proven-strengthened, not an inferred default, and no
-- historical winner was ever attested this way. That is intentional, not
-- an oversight: retroactively re-attesting existing schema-7/8 data (or
-- quarantining what can't be re-attested) is HI128's job, not this
-- migration's -- HI127 only decides the representation.
--
-- Per-winner granularity, not per-(build,signature): inventory.py's own
-- `INSERT OR REPLACE INTO winner` does not include winner_id in its column
-- list, so a conflict on winner's own UNIQUE(build_id, hardware_id,
-- objective, dispatch_digest) constraint deletes the old row and inserts a
-- genuinely NEW one with a new winner_id -- a (build,signature)-keyed
-- marker would NOT be invalidated by that replace and could wrongly
-- attest the new (possibly unverified) row. ON DELETE CASCADE below closes
-- that automatically: destroying the old winner row destroys its stale
-- attestation with it.
--
-- This migration is applied via Python's sqlite3.Connection.executescript()
-- (see tools/tests/campaign/test_db_migration.py for the established real
-- invocation pattern for this project's migrations) -- executescript()
-- stops at the first statement that raises, so the schema-version guard
-- below (a CHECK-constraint violation on a temp table) genuinely aborts the
-- whole script rather than silently continuing on a version mismatch.

PRAGMA foreign_keys = ON;

BEGIN;

-- Fail closed unless this database is currently EXACTLY schema 7. A CHECK
-- violation here is a real SQLite error that halts the rest of this script.
CREATE TEMP TABLE _hi127_schema_guard (
    version TEXT NOT NULL CHECK (version = '7')
);

INSERT INTO _hi127_schema_guard(version)
VALUES ((SELECT value FROM schema_meta WHERE key = 'schema_version'));

DROP TABLE _hi127_schema_guard;

CREATE TABLE winner_verification (
    winner_id             INTEGER PRIMARY KEY
                           REFERENCES winner(winner_id) ON DELETE CASCADE,
    verification_profile  TEXT    NOT NULL,
    verified_at           TEXT    NOT NULL DEFAULT (datetime('now')),
    CHECK (verification_profile = 'hi121-strengthened-ingest-v1')
);

UPDATE schema_meta SET value = '8' WHERE key = 'schema_version';

-- Fail closed on any foreign-key violation introduced by this migration,
-- BEFORE committing -- same pattern as 0007_producer_capabilities.sql.
CREATE TEMP TABLE _hi127_fk_guard (x INTEGER CHECK (0));

INSERT INTO _hi127_fk_guard SELECT 1 FROM pragma_foreign_key_check();

DROP TABLE _hi127_fk_guard;

COMMIT;
