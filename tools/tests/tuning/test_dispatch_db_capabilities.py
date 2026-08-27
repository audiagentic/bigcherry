"""HI121 M2: tests for schema-7's build_capability table, the winner-index
fix, and the 0006->0007 migration."""

from __future__ import annotations

import sqlite3
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_SQL = REPO_ROOT / "sql" / "dispatch-db.sql"
MIGRATION_0007 = REPO_ROOT / "sql" / "migrations" / "0007_producer_capabilities.sql"


def _fresh_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA_SQL.read_text(encoding="utf-8"))
    return conn


def _insert_build(conn: sqlite3.Connection, *, manifest_hash: str) -> int:
    conn.execute(
        "INSERT INTO build (source_revision, manifest_hash, signature_schema, "
        "hardware_schema, variant_set) VALUES ('deadbeefdeadbeefdead', ?, 2, 1, 'inventory')",
        (manifest_hash,),
    )
    return conn.execute("SELECT last_insert_rowid()").fetchone()[0]


class FreshSchemaTests(unittest.TestCase):
    def test_schema_version_is_8(self):
        conn = _fresh_connection()
        value = conn.execute("SELECT value FROM schema_meta WHERE key='schema_version'").fetchone()[0]
        self.assertEqual(value, "8")

    def test_build_capability_table_exists_with_check_constraint(self):
        conn = _fresh_connection()
        build_id = _insert_build(conn, manifest_hash="aa")
        conn.execute(
            "INSERT INTO build_capability (build_id, backend, producer_capabilities) VALUES (?, 'hip', ?)",
            (build_id, bytes.fromhex("0000000000000000000000000000001f")),
        )
        conn.commit()
        row = conn.execute(
            "SELECT producer_capabilities FROM build_capability WHERE build_id=? AND backend='hip'",
            (build_id,),
        ).fetchone()
        self.assertEqual(row[0], bytes.fromhex("0000000000000000000000000000001f"))

    def test_build_capability_rejects_wrong_length_blob_too_short(self):
        conn = _fresh_connection()
        build_id = _insert_build(conn, manifest_hash="aa")
        with self.assertRaises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO build_capability (build_id, backend, producer_capabilities) VALUES (?, 'hip', ?)",
                (build_id, b"\x00" * 15),
            )

    def test_build_capability_rejects_wrong_length_blob_too_long(self):
        conn = _fresh_connection()
        build_id = _insert_build(conn, manifest_hash="aa")
        with self.assertRaises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO build_capability (build_id, backend, producer_capabilities) VALUES (?, 'hip', ?)",
                (build_id, b"\x00" * 17),
            )

    def test_two_backends_can_coexist_for_the_same_build(self):
        conn = _fresh_connection()
        build_id = _insert_build(conn, manifest_hash="aa")
        conn.execute(
            "INSERT INTO build_capability (build_id, backend, producer_capabilities) VALUES (?, 'hip', ?)",
            (build_id, b"\x00" * 16),
        )
        conn.execute(
            "INSERT INTO build_capability (build_id, backend, producer_capabilities) VALUES (?, 'vulkan', ?)",
            (build_id, b"\x01" + b"\x00" * 15),
        )
        conn.commit()
        count = conn.execute(
            "SELECT COUNT(*) FROM build_capability WHERE build_id=?", (build_id,)
        ).fetchone()[0]
        self.assertEqual(count, 2)


class WinnerIndexFixTests(unittest.TestCase):
    def _insert_winner(self, conn, *, build_id, hardware_id, candidate_id, dispatch_hex):
        conn.execute(
            "INSERT INTO winner (build_id, hardware_id, candidate_id, objective, "
            "dispatch_digest, stable_name, native_stable_name, is_native, "
            "improvement_pct, median_us, p95_us) VALUES "
            "(?, ?, ?, 'latency', ?, 'native', 'native', 1, 0.0, 1.0, 1.0)",
            (build_id, hardware_id, candidate_id, bytes.fromhex(dispatch_hex)),
        )

    def test_two_different_builds_can_both_retain_a_winner_for_the_same_dispatch(self):
        conn = _fresh_connection()
        build_a = _insert_build(conn, manifest_hash="aa")
        build_b = _insert_build(conn, manifest_hash="bb")
        conn.execute(
            "INSERT INTO hardware (hardware_digest, architecture, architecture_code, "
            "wave_size, compute_units, feature_flags, canonical_json) VALUES "
            "(?, 'gfx1100', 1, 32, 96, 0, '{}')",
            (bytes.fromhex("11" * 16),),
        )
        hardware_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        for build_id in (build_a, build_b):
            conn.execute(
                "INSERT INTO candidate (build_id, stable_name, family, source_class, "
                "implementation_version, architectures, architecture_mask, graph_safe, "
                "deterministic, config_json) VALUES (?, 'native', 'mmvq', 'native_wrapper', "
                "1, '[]', 0, 1, 1, '{}')",
                (build_id,),
            )
        candidate_a = conn.execute(
            "SELECT candidate_id FROM candidate WHERE build_id=?", (build_a,)
        ).fetchone()[0]
        candidate_b = conn.execute(
            "SELECT candidate_id FROM candidate WHERE build_id=?", (build_b,)
        ).fetchone()[0]
        dispatch_hex = "22" * 16
        self._insert_winner(conn, build_id=build_a, hardware_id=hardware_id, candidate_id=candidate_a, dispatch_hex=dispatch_hex)
        # Before the fix this raised sqlite3.IntegrityError on the global
        # UNIQUE(dispatch_digest, objective) index, even though this is a
        # genuinely different build.
        self._insert_winner(conn, build_id=build_b, hardware_id=hardware_id, candidate_id=candidate_b, dispatch_hex=dispatch_hex)
        conn.commit()
        count = conn.execute(
            "SELECT COUNT(*) FROM winner WHERE dispatch_digest=?", (bytes.fromhex(dispatch_hex),)
        ).fetchone()[0]
        self.assertEqual(count, 2)


class Migration0007Tests(unittest.TestCase):
    def _schema6_fixture(self) -> sqlite3.Connection:
        conn = sqlite3.connect(":memory:")
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(
            """
            CREATE TABLE schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            INSERT INTO schema_meta VALUES ('schema_version', '6');
            CREATE TABLE build (build_id INTEGER PRIMARY KEY);
            CREATE TABLE winner (
                winner_id INTEGER PRIMARY KEY,
                build_id INTEGER NOT NULL,
                dispatch_digest BLOB NOT NULL,
                objective TEXT NOT NULL
            );
            CREATE UNIQUE INDEX winner_dispatch_idx ON winner(dispatch_digest, objective);
            CREATE TABLE vk_winner (
                vk_winner_id INTEGER PRIMARY KEY,
                build_id INTEGER NOT NULL,
                dispatch_digest BLOB NOT NULL,
                objective TEXT NOT NULL
            );
            CREATE UNIQUE INDEX vk_winner_dispatch_idx ON vk_winner(dispatch_digest, objective);
            """
        )
        return conn

    def test_migration_upgrades_schema6_to_schema7(self):
        conn = self._schema6_fixture()
        conn.executescript(MIGRATION_0007.read_text(encoding="utf-8"))
        value = conn.execute("SELECT value FROM schema_meta WHERE key='schema_version'").fetchone()[0]
        self.assertEqual(value, "7")
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        self.assertIn("build_capability", tables)
        # Index is no longer unique: two rows sharing (dispatch_digest, objective)
        # across different builds must now be insertable.
        conn.execute("INSERT INTO build (build_id) VALUES (1), (2)")
        conn.execute("INSERT INTO winner (build_id, dispatch_digest, objective) VALUES (1, x'aa', 'latency')")
        conn.execute("INSERT INTO winner (build_id, dispatch_digest, objective) VALUES (2, x'aa', 'latency')")
        conn.commit()
        count = conn.execute("SELECT COUNT(*) FROM winner WHERE dispatch_digest=x'aa'").fetchone()[0]
        self.assertEqual(count, 2)

    def test_migration_does_not_backfill_build_capability(self):
        conn = self._schema6_fixture()
        conn.execute("INSERT INTO build (build_id) VALUES (1)")
        conn.commit()
        conn.executescript(MIGRATION_0007.read_text(encoding="utf-8"))
        count = conn.execute("SELECT COUNT(*) FROM build_capability").fetchone()[0]
        self.assertEqual(count, 0)

    def test_migration_fails_closed_against_a_database_not_at_schema_6(self):
        conn = self._schema6_fixture()
        conn.execute("UPDATE schema_meta SET value='5' WHERE key='schema_version'")
        conn.commit()
        with self.assertRaises(sqlite3.IntegrityError):
            conn.executescript(MIGRATION_0007.read_text(encoding="utf-8"))
        # Confirm no partial application: schema_version untouched, no
        # build_capability table created.
        value = conn.execute("SELECT value FROM schema_meta WHERE key='schema_version'").fetchone()[0]
        self.assertEqual(value, "5")
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        self.assertNotIn("build_capability", tables)

    def test_migration_fails_closed_when_run_again_against_an_already_v7_database(self):
        conn = self._schema6_fixture()
        conn.executescript(MIGRATION_0007.read_text(encoding="utf-8"))
        # Re-running must fail rather than silently double-apply. The
        # schema-version guard itself catches this first (the database is
        # now at 7, not 6) -- it never even reaches the build_capability
        # table creation the second time.
        with self.assertRaises(sqlite3.IntegrityError):
            conn.executescript(MIGRATION_0007.read_text(encoding="utf-8"))
        value = conn.execute("SELECT value FROM schema_meta WHERE key='schema_version'").fetchone()[0]
        self.assertEqual(value, "7")


if __name__ == "__main__":
    unittest.main()
