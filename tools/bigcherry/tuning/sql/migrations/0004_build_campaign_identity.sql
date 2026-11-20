-- BigCherry dispatch DB schema 3 -> 4.
--
-- The build table's row identity did not match the identity model: the old
-- table-level UNIQUE (source_revision, manifest_hash, signature_schema,
-- hardware_schema, variant_set, build_descriptor_hash) applied to every
-- row regardless of whether it carried a real campaign identity, so two
-- campaign builds sharing that legacy key but genuinely different campaign
-- identity could not both exist as distinct rows. This migration adds
-- identity_scope ('campaign' | 'legacy-imported') and replaces the single
-- table-level UNIQUE with two partial unique indexes, one per scope.
--
-- SQLite cannot drop or alter a table-level UNIQUE constraint in place --
-- this requires a full table rebuild (the documented SQLite ALTER TABLE
-- procedure), not just ALTER TABLE ADD COLUMN as schema 2->3 used.
--
-- ONLY rows carrying the COMPLETE campaign identity (source_slice_id AND
-- build_plan_id AND effective_build_id all non-null) become 'campaign';
-- every other row -- including one with a PARTIAL campaign identity --
-- becomes 'legacy-imported'. A partial identity is not campaign evidence;
-- silently promoting it would misrepresent what is actually known about
-- that row.

PRAGMA foreign_keys = OFF;

BEGIN;

CREATE TABLE build_v4 (
    build_id            INTEGER PRIMARY KEY,
    source_revision     TEXT    NOT NULL,
    source_dirty        INTEGER NOT NULL DEFAULT 0,
    manifest_hash       TEXT    NOT NULL,
    bigcherry_revision  TEXT,
    signature_schema    INTEGER NOT NULL,
    hardware_schema     INTEGER NOT NULL,
    variant_set         TEXT    NOT NULL,
    rocm_version        TEXT,
    hip_version         TEXT,
    compiler            TEXT,
    build_descriptor_hash TEXT,
    source_slice_id      TEXT,
    build_plan_id        TEXT,
    effective_build_id   TEXT,
    campaign_run_id      TEXT,
    workload_id          TEXT,
    dispatch_abi        TEXT,
    identity_scope       TEXT    NOT NULL DEFAULT 'legacy-imported',
    created_at          TEXT    NOT NULL DEFAULT (datetime('now')),
    CHECK (identity_scope IN ('campaign', 'legacy-imported')),
    CHECK (identity_scope = 'legacy-imported' OR (
        source_slice_id IS NOT NULL
        AND build_plan_id IS NOT NULL
        AND effective_build_id IS NOT NULL
    ))
);

INSERT INTO build_v4 (
    build_id, source_revision, source_dirty, manifest_hash, bigcherry_revision,
    signature_schema, hardware_schema, variant_set, rocm_version, hip_version,
    compiler, build_descriptor_hash, source_slice_id, build_plan_id,
    effective_build_id, campaign_run_id, workload_id, dispatch_abi,
    identity_scope, created_at
)
SELECT
    build_id, source_revision, source_dirty, manifest_hash, bigcherry_revision,
    signature_schema, hardware_schema, variant_set, rocm_version, hip_version,
    compiler, build_descriptor_hash, source_slice_id, build_plan_id,
    effective_build_id, campaign_run_id, workload_id, dispatch_abi,
    CASE
        WHEN source_slice_id IS NOT NULL
         AND build_plan_id IS NOT NULL
         AND effective_build_id IS NOT NULL
        THEN 'campaign'
        ELSE 'legacy-imported'
    END,
    created_at
FROM build;

DROP TABLE build;
ALTER TABLE build_v4 RENAME TO build;

CREATE UNIQUE INDEX IF NOT EXISTS build_campaign_identity_uq
    ON build(source_slice_id, build_plan_id, effective_build_id)
    WHERE identity_scope = 'campaign';

CREATE UNIQUE INDEX IF NOT EXISTS build_legacy_identity_uq
    ON build(source_revision, manifest_hash, signature_schema,
             hardware_schema, variant_set, build_descriptor_hash)
    WHERE identity_scope = 'legacy-imported';

UPDATE schema_meta SET value = '4' WHERE key = 'schema_version';

COMMIT;

PRAGMA foreign_keys = ON;
PRAGMA foreign_key_check;
