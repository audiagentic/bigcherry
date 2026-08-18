"""RE09/RV50 schema-4: the build table's real identity_scope boundary.

Real SQLite throughout (no mocking): a real schema-3 fixture DB (built the
same way schema 2->3 rows would look, with real FK children in
observation/measurement/winner/replay_miss), migrated via
sql/migrations/0004_build_campaign_identity.sql, then PRAGMA
foreign_key_check proving nothing broke.
"""

from __future__ import annotations

import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_V3_BASE = REPO_ROOT / "sql" / "dispatch-db.sql"
MIGRATION_0004 = REPO_ROOT / "sql" / "migrations" / "0004_build_campaign_identity.sql"


def _schema_v3_ddl() -> str:
    """Reconstruct the schema-3 build table shape (pre-migration) directly
    from the recorded 0003 migration + a hand-written base -- the base
    dispatch-db.sql file now IS schema-4, so a schema-3 fixture has to be
    built explicitly rather than read off disk."""
    return """
    PRAGMA foreign_keys = ON;
    CREATE TABLE schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
    INSERT INTO schema_meta(key, value) VALUES ('schema_version', '3');

    CREATE TABLE build (
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
        created_at          TEXT    NOT NULL DEFAULT (datetime('now')),
        UNIQUE (source_revision, manifest_hash, signature_schema,
                hardware_schema, variant_set, build_descriptor_hash)
    );

    CREATE TABLE hardware (
        hardware_id       INTEGER PRIMARY KEY,
        hardware_digest   BLOB    NOT NULL UNIQUE,
        architecture      TEXT    NOT NULL,
        architecture_code INTEGER NOT NULL,
        wave_size         INTEGER NOT NULL,
        compute_units     INTEGER NOT NULL,
        feature_flags     INTEGER NOT NULL,
        canonical_json    TEXT    NOT NULL,
        created_at        TEXT    NOT NULL DEFAULT (datetime('now'))
    );

    CREATE TABLE signature (
        signature_id     INTEGER PRIMARY KEY,
        signature_digest BLOB    NOT NULL UNIQUE,
        base_digest      BLOB    NOT NULL,
        schema_version   INTEGER NOT NULL,
        op               TEXT    NOT NULL,
        src0_type        TEXT    NOT NULL,
        src1_type        TEXT    NOT NULL,
        dst_type         TEXT    NOT NULL,
        m INTEGER NOT NULL, n INTEGER NOT NULL, k INTEGER NOT NULL,
        fusion           TEXT    NOT NULL DEFAULT 'none',
        is_refined       INTEGER NOT NULL DEFAULT 0,
        canonical_json   TEXT    NOT NULL,
        created_at       TEXT    NOT NULL DEFAULT (datetime('now'))
    );

    CREATE TABLE candidate (
        candidate_id           INTEGER PRIMARY KEY,
        build_id               INTEGER NOT NULL REFERENCES build(build_id),
        stable_name            TEXT    NOT NULL,
        family                 TEXT    NOT NULL,
        source_class           TEXT    NOT NULL,
        implementation_version INTEGER NOT NULL,
        architectures          TEXT    NOT NULL,
        architecture_mask      INTEGER NOT NULL,
        graph_safe             INTEGER NOT NULL,
        deterministic          INTEGER NOT NULL,
        config_json            TEXT    NOT NULL,
        UNIQUE (build_id, stable_name)
    );

    CREATE TABLE observation (
        observation_id      INTEGER PRIMARY KEY,
        build_id            INTEGER NOT NULL REFERENCES build(build_id),
        hardware_id         INTEGER NOT NULL REFERENCES hardware(hardware_id),
        signature_id        INTEGER NOT NULL REFERENCES signature(signature_id),
        native_stable_name  TEXT    NOT NULL,
        calls               INTEGER NOT NULL DEFAULT 0,
        est_bytes           INTEGER NOT NULL DEFAULT 0,
        est_flops           INTEGER NOT NULL DEFAULT 0,
        sites_json          TEXT    NOT NULL DEFAULT '[]',
        diagnostics_json    TEXT    NOT NULL DEFAULT '{}',
        source_slice_id     TEXT,
        workload_id         TEXT,
        campaign_run_id     TEXT,
        first_seen          TEXT    NOT NULL DEFAULT (datetime('now')),
        last_seen           TEXT    NOT NULL DEFAULT (datetime('now')),
        UNIQUE (build_id, hardware_id, signature_id)
    );

    CREATE TABLE tuning_run (
        run_id                 INTEGER PRIMARY KEY,
        build_id               INTEGER NOT NULL REFERENCES build(build_id),
        run_digest             BLOB    NOT NULL UNIQUE,
        workload_digest        BLOB,
        workload_label         TEXT,
        started_at TEXT, finished_at TEXT, host_sync_overhead_us REAL,
        config_json  TEXT NOT NULL DEFAULT '{}',
        machine_json TEXT NOT NULL DEFAULT '{}',
        created_at   TEXT NOT NULL DEFAULT (datetime('now'))
    );

    CREATE TABLE measurement (
        measurement_id   INTEGER PRIMARY KEY,
        build_id         INTEGER NOT NULL REFERENCES build(build_id),
        hardware_id      INTEGER NOT NULL REFERENCES hardware(hardware_id),
        signature_id     INTEGER REFERENCES signature(signature_id),
        dispatch_digest  BLOB,
        candidate_id     INTEGER NOT NULL REFERENCES candidate(candidate_id),
        run_id           INTEGER REFERENCES tuning_run(run_id),
        objective        TEXT NOT NULL DEFAULT 'latency',
        stage            TEXT NOT NULL,
        accepted         INTEGER NOT NULL,
        reject_reason    TEXT,
        samples          INTEGER NOT NULL DEFAULT 0,
        launches_per_sample INTEGER,
        median_us REAL, gpu_mad_us REAL, p95_us REAL, host_median_us REAL,
        min_us REAL, stddev_us REAL,
        workspace_bytes  INTEGER NOT NULL DEFAULT 0,
        pool_peak_bytes  INTEGER,
        nmse REAL, max_abs_err REAL, max_rel_err REAL,
        samples_json TEXT, effective_us REAL,
        source_slice_id TEXT, build_plan_id TEXT, effective_build_id TEXT,
        workload_id TEXT, campaign_run_id TEXT,
        measured_at TEXT NOT NULL DEFAULT (datetime('now')),
        UNIQUE (build_id, hardware_id, candidate_id, objective, stage, dispatch_digest)
    );

    CREATE TABLE winner (
        winner_id           INTEGER PRIMARY KEY,
        build_id            INTEGER NOT NULL REFERENCES build(build_id),
        hardware_id         INTEGER NOT NULL REFERENCES hardware(hardware_id),
        signature_id        INTEGER REFERENCES signature(signature_id),
        objective            TEXT NOT NULL DEFAULT 'latency',
        dispatch_digest      BLOB NOT NULL,
        candidate_id         INTEGER NOT NULL REFERENCES candidate(candidate_id),
        run_id               INTEGER REFERENCES tuning_run(run_id),
        stable_name           TEXT NOT NULL,
        native_stable_name    TEXT NOT NULL,
        is_native             INTEGER NOT NULL,
        improvement_pct       REAL NOT NULL DEFAULT 0.0,
        median_us REAL NOT NULL, p95_us REAL NOT NULL,
        workspace_bytes INTEGER NOT NULL DEFAULT 0,
        pool_peak_bytes INTEGER,
        reason TEXT, confidence REAL,
        seeded INTEGER NOT NULL DEFAULT 0,
        validated INTEGER NOT NULL DEFAULT 0,
        promotion_status TEXT, q_value REAL,
        source_slice_id TEXT, build_plan_id TEXT, effective_build_id TEXT,
        workload_id TEXT, campaign_run_id TEXT,
        decided_at TEXT NOT NULL DEFAULT (datetime('now')),
        UNIQUE (build_id, hardware_id, objective, dispatch_digest)
    );

    CREATE TABLE replay_miss (
        miss_id          INTEGER PRIMARY KEY,
        build_id         INTEGER NOT NULL REFERENCES build(build_id),
        hardware_id      INTEGER NOT NULL REFERENCES hardware(hardware_id),
        signature_digest BLOB NOT NULL,
        dispatch_digest  BLOB NOT NULL,
        canonical_json   TEXT NOT NULL,
        fallback_name    TEXT NOT NULL,
        source_slice_id  TEXT, workload_id TEXT, campaign_run_id TEXT,
        calls            INTEGER NOT NULL DEFAULT 1,
        first_seen TEXT NOT NULL DEFAULT (datetime('now')),
        last_seen  TEXT NOT NULL DEFAULT (datetime('now')),
        UNIQUE (build_id, hardware_id, dispatch_digest)
    );
    """


def _seed_schema_v3_fixture(connection: sqlite3.Connection) -> None:
    """One legacy-only build row and one campaign-identified build row,
    each with real FK children in every dependent table -- exactly the
    shape the migration must carry through intact."""
    connection.executescript(_schema_v3_ddl())
    connection.execute(
        "INSERT INTO build (build_id, source_revision, manifest_hash, "
        "signature_schema, hardware_schema, variant_set, build_descriptor_hash) "
        "VALUES (1, 'rev-legacy', 'manifest-legacy', 1, 1, 'inventory', 'descriptor-legacy')"
    )
    connection.execute(
        "INSERT INTO build (build_id, source_revision, manifest_hash, "
        "signature_schema, hardware_schema, variant_set, build_descriptor_hash, "
        "source_slice_id, build_plan_id, effective_build_id, campaign_run_id, workload_id) "
        "VALUES (2, 'rev-campaign', 'manifest-campaign', 1, 1, 'workload-max', 'descriptor-campaign', "
        "'slice-1', 'plan-1', 'effective-1', 'run-1', 'workload-1')"
    )
    connection.execute(
        "INSERT INTO hardware (hardware_id, hardware_digest, architecture, "
        "architecture_code, wave_size, compute_units, feature_flags, canonical_json) "
        "VALUES (1, X'00', 'gfx1100', 1, 32, 60, 0, '{}')"
    )
    connection.execute(
        "INSERT INTO signature (signature_id, signature_digest, base_digest, "
        "schema_version, op, src0_type, src1_type, dst_type, m, n, k, canonical_json) "
        "VALUES (1, X'01', X'01', 1, 'MUL_MAT', 'f16', 'f32', 'f32', 1, 1, 1, '{}')"
    )
    for build_id in (1, 2):
        connection.execute(
            "INSERT INTO candidate (build_id, stable_name, family, source_class, "
            "implementation_version, architectures, architecture_mask, graph_safe, "
            "deterministic, config_json) "
            f"VALUES ({build_id}, 'mmq:native:v1', 'mmq', 'native_wrapper', 1, '[]', 0, 1, 1, '{{}}')"
        )
        connection.execute(
            "INSERT INTO observation (build_id, hardware_id, signature_id, "
            "native_stable_name) VALUES (?, 1, 1, 'mmq:native:v1')", (build_id,)
        )
        connection.execute(
            "INSERT INTO measurement (build_id, hardware_id, signature_id, "
            "candidate_id, stage, accepted) "
            "SELECT ?, 1, 1, candidate_id, 'final', 1 FROM candidate WHERE build_id = ?",
            (build_id, build_id),
        )
        connection.execute(
            "INSERT INTO winner (build_id, hardware_id, dispatch_digest, "
            "candidate_id, stable_name, native_stable_name, is_native, "
            "median_us, p95_us) "
            "SELECT ?, 1, X'02', candidate_id, 'mmq:native:v1', 'mmq:native:v1', 1, 1.0, 1.0 "
            "FROM candidate WHERE build_id = ?",
            (build_id, build_id),
        )
        connection.execute(
            "INSERT INTO replay_miss (build_id, hardware_id, signature_digest, "
            "dispatch_digest, canonical_json, fallback_name) "
            "VALUES (?, 1, X'01', X'03', '{}', 'mmq:native:v1')", (build_id,)
        )
    connection.commit()


class SchemaV3ToV4MigrationTests(unittest.TestCase):
    def _migrated_connection(self) -> sqlite3.Connection:
        db_path = Path(tempfile.mkstemp(suffix=".sqlite")[1])
        connection = sqlite3.connect(str(db_path))
        _seed_schema_v3_fixture(connection)
        connection.executescript(MIGRATION_0004.read_text(encoding="utf-8"))
        return connection

    def test_migration_derives_identity_scope_correctly(self):
        connection = self._migrated_connection()
        try:
            rows = dict(connection.execute(
                "SELECT build_id, identity_scope FROM build").fetchall())
            self.assertEqual(rows, {1: "legacy-imported", 2: "campaign"})
        finally:
            connection.close()

    def test_migration_preserves_every_fk_child_row(self):
        connection = self._migrated_connection()
        try:
            for table in ("candidate", "observation", "measurement", "winner", "replay_miss"):
                count = connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                self.assertEqual(count, 2, f"{table} lost rows across the migration")
        finally:
            connection.close()

    def test_migration_bumps_schema_version_and_passes_foreign_key_check(self):
        connection = self._migrated_connection()
        try:
            version = connection.execute(
                "SELECT value FROM schema_meta WHERE key = 'schema_version'").fetchone()[0]
            self.assertEqual(version, "4")
            connection.execute("PRAGMA foreign_keys = ON")
            violations = connection.execute("PRAGMA foreign_key_check").fetchall()
            self.assertEqual(violations, [])
        finally:
            connection.close()

    def test_partial_campaign_identity_becomes_legacy_imported_not_campaign(self):
        # A row with only SOME of the campaign-identity columns filled
        # must not be silently promoted to 'campaign' -- that would
        # misrepresent what is actually known about it.
        db_path = Path(tempfile.mkstemp(suffix=".sqlite")[1])
        connection = sqlite3.connect(str(db_path))
        try:
            connection.executescript(_schema_v3_ddl())
            connection.execute(
                "INSERT INTO build (build_id, source_revision, manifest_hash, "
                "signature_schema, hardware_schema, variant_set, build_descriptor_hash, "
                "source_slice_id, build_plan_id) "  # effective_build_id missing
                "VALUES (1, 'rev', 'manifest', 1, 1, 'inventory', 'descriptor', 'slice-1', 'plan-1')"
            )
            connection.commit()
            connection.executescript(MIGRATION_0004.read_text(encoding="utf-8"))
            scope = connection.execute(
                "SELECT identity_scope FROM build WHERE build_id = 1").fetchone()[0]
            self.assertEqual(scope, "legacy-imported")
        finally:
            connection.close()

    def test_migrated_schema_still_enforces_both_partial_unique_indexes(self):
        connection = self._migrated_connection()
        try:
            # A second row claiming build_id 2's exact campaign triple fails.
            with self.assertRaises(sqlite3.IntegrityError):
                connection.execute(
                    "INSERT INTO build (source_revision, manifest_hash, "
                    "signature_schema, hardware_schema, variant_set, "
                    "source_slice_id, build_plan_id, effective_build_id, identity_scope) "
                    "VALUES ('other-rev', 'other-manifest', 1, 1, 'inventory', "
                    "'slice-1', 'plan-1', 'effective-1', 'campaign')"
                )
            # A second row claiming build_id 1's exact legacy key fails too.
            with self.assertRaises(sqlite3.IntegrityError):
                connection.execute(
                    "INSERT INTO build (source_revision, manifest_hash, "
                    "signature_schema, hardware_schema, variant_set, "
                    "build_descriptor_hash, identity_scope) "
                    "VALUES ('rev-legacy', 'manifest-legacy', 1, 1, 'inventory', "
                    "'descriptor-legacy', 'legacy-imported')"
                )
        finally:
            connection.close()


class ReplayPortableFormatUnaffectedTests(unittest.TestCase):
    """RE09/RV50: the portable replay dispatch key must never depend on
    campaign DB identity (standards 9.1's own 'PORTABLE REPLAY IDENTITY
    CLEAN' rule) -- proven at the code level: replay_cache.py's own
    queries never reference any campaign-identity or identity_scope
    column at all, so this schema change cannot leak into the wire
    format by construction, not merely by omission in this one test."""

    def test_replay_cache_module_never_references_campaign_identity_columns(self):
        source = (REPO_ROOT / "tools" / "bigcherry" / "replay_cache.py").read_text(encoding="utf-8")
        for forbidden in (
            "identity_scope", "campaign_run_id", "source_slice_id",
            "build_plan_id", "effective_build_id",
        ):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
