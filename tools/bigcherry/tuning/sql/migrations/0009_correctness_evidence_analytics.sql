-- BigCherry dispatch DB schema 8 -> 9 (HTR01, adversarially designed with
-- GPT, session ses_330ae3c055084f38, 2026-08-30).
--
-- Real-hardware validation of HTR01's recovery search found that every
-- already-measured alternative candidate lacked correctness evidence
-- (normal campaigns only evidence the eventual winner per signature), so
-- recovery degraded to safe-but-useless all-native fallback. The fix is
-- lazy, on-demand evidence generation for exactly the alternative recovery
-- is about to try -- but that lazily-generated evidence should remain a
-- first-class, reusable dataset (numerical-family clustering, cross-
-- signature pattern analysis), not just a disposable pass/fail check.
--
-- This migration adds:
--   1. correctness_evidence_origin -- WHY a correctness_evidence row
--      exists (promotion_winner | recovery_alternative | manual_analysis).
--      Existing rows get NO origin row at all (not a NULL reason -- no
--      row means "predates origin tracking", never an inferred default).
--   2. Four nullable columns on correctness_evidence_seed capturing the
--      EXACT backend output byte digests the patched producer already
--      emits (HI83's backend1_digest/backend2_digest) but which were
--      previously parsed and discarded -- enabling exact numerical-family
--      clustering, a strictly stronger signal than matching NMSE alone.
--      Existing seed rows get these four columns as NULL.
--
-- Applied via Python's sqlite3.Connection.executescript() (see
-- tools/tests/campaign/test_db_migration_0008.py for the established real
-- invocation pattern) -- executescript() stops at the first statement that
-- raises, so the schema-version guard below (a CHECK-constraint violation
-- on a temp table) genuinely aborts the whole script rather than silently
-- continuing on a version mismatch.

PRAGMA foreign_keys = ON;

BEGIN;

-- Fail closed unless this database is currently EXACTLY schema 8.
CREATE TEMP TABLE _htr01_schema_guard (
    version TEXT NOT NULL CHECK (version = '8')
);

INSERT INTO _htr01_schema_guard(version)
VALUES ((SELECT value FROM schema_meta WHERE key = 'schema_version'));

DROP TABLE _htr01_schema_guard;

ALTER TABLE correctness_evidence_seed ADD COLUMN native_output_digest TEXT;
ALTER TABLE correctness_evidence_seed ADD COLUMN candidate_output_digest TEXT;
ALTER TABLE correctness_evidence_seed ADD COLUMN reference_output_digest TEXT;
ALTER TABLE correctness_evidence_seed ADD COLUMN output_nels INTEGER;

CREATE TABLE correctness_evidence_origin (
    correctness_evidence_id INTEGER PRIMARY KEY
        REFERENCES correctness_evidence(correctness_evidence_id) ON DELETE CASCADE,
    reason           TEXT NOT NULL,
    campaign_run_id  TEXT,
    recovery_run_id  TEXT,
    created_at       TEXT NOT NULL DEFAULT (datetime('now')),
    CHECK (reason IN ('promotion_winner', 'recovery_alternative', 'manual_analysis'))
);

UPDATE schema_meta SET value = '9' WHERE key = 'schema_version';

-- Fail closed on any foreign-key violation introduced by this migration,
-- BEFORE committing -- same pattern as 0007/0008.
CREATE TEMP TABLE _htr01_fk_guard (x INTEGER CHECK (0));

INSERT INTO _htr01_fk_guard SELECT 1 FROM pragma_foreign_key_check();

DROP TABLE _htr01_fk_guard;

COMMIT;
