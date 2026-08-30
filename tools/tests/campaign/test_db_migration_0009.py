"""HTR01: schema-8 to schema-9 correctness_evidence_origin + exact output
digest migration. Follows the same minimal-fixture-DB convention as
test_db_migration_0008.py."""

from __future__ import annotations

import sqlite3
import unittest
from pathlib import Path

MIGRATION = Path(__file__).resolve().parents[3] / "sql" / "migrations" / "0009_correctness_evidence_analytics.sql"


class Migration0009Tests(unittest.TestCase):
    def _fixture_connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(":memory:")
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript(
            """
            CREATE TABLE schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            INSERT INTO schema_meta VALUES ('schema_version', '8');
            CREATE TABLE correctness_evidence (
                correctness_evidence_id INTEGER PRIMARY KEY
            );
            INSERT INTO correctness_evidence (correctness_evidence_id) VALUES (1);
            CREATE TABLE correctness_evidence_seed (
                correctness_evidence_seed_id INTEGER PRIMARY KEY,
                correctness_evidence_id INTEGER NOT NULL,
                seed INTEGER NOT NULL
            );
            INSERT INTO correctness_evidence_seed
                (correctness_evidence_seed_id, correctness_evidence_id, seed)
                VALUES (1, 1, 1);
            """
        )
        return connection

    def test_migration_adds_origin_table_and_seed_columns_and_bumps_schema(self):
        connection = self._fixture_connection()
        connection.executescript(MIGRATION.read_text(encoding="utf-8"))
        self.assertEqual(
            connection.execute("SELECT value FROM schema_meta WHERE key='schema_version'").fetchone()[0],
            "9",
        )
        seed_columns = {row[1] for row in connection.execute("PRAGMA table_info(correctness_evidence_seed)")}
        self.assertTrue(
            {"native_output_digest", "candidate_output_digest", "reference_output_digest", "output_nels"}
            <= seed_columns
        )
        origin_columns = {row[1] for row in connection.execute("PRAGMA table_info(correctness_evidence_origin)")}
        self.assertTrue({"correctness_evidence_id", "reason", "campaign_run_id", "recovery_run_id"} <= origin_columns)

    def test_migration_refuses_wrong_starting_schema_version(self):
        connection = sqlite3.connect(":memory:")
        connection.executescript(
            """
            CREATE TABLE schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            INSERT INTO schema_meta VALUES ('schema_version', '7');
            CREATE TABLE correctness_evidence (correctness_evidence_id INTEGER PRIMARY KEY);
            CREATE TABLE correctness_evidence_seed (
                correctness_evidence_seed_id INTEGER PRIMARY KEY,
                correctness_evidence_id INTEGER NOT NULL,
                seed INTEGER NOT NULL
            );
            """
        )
        with self.assertRaises(sqlite3.IntegrityError):
            connection.executescript(MIGRATION.read_text(encoding="utf-8"))

    def test_existing_evidence_gets_no_origin_row_not_a_null_reason(self):
        connection = self._fixture_connection()
        connection.executescript(MIGRATION.read_text(encoding="utf-8"))
        count = connection.execute("SELECT COUNT(*) FROM correctness_evidence_origin").fetchone()[0]
        self.assertEqual(count, 0)

    def test_only_known_reason_values_accepted(self):
        connection = self._fixture_connection()
        connection.executescript(MIGRATION.read_text(encoding="utf-8"))
        with self.assertRaises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO correctness_evidence_origin (correctness_evidence_id, reason) "
                "VALUES (1, 'not_a_real_reason')"
            )

    def test_deleting_evidence_row_cascades_to_origin(self):
        connection = self._fixture_connection()
        connection.executescript(MIGRATION.read_text(encoding="utf-8"))
        connection.execute(
            "INSERT INTO correctness_evidence_origin (correctness_evidence_id, reason) "
            "VALUES (1, 'recovery_alternative')"
        )
        connection.commit()
        connection.execute("DELETE FROM correctness_evidence WHERE correctness_evidence_id = 1")
        connection.commit()
        remaining = connection.execute("SELECT COUNT(*) FROM correctness_evidence_origin").fetchone()[0]
        self.assertEqual(remaining, 0)


if __name__ == "__main__":
    unittest.main()
