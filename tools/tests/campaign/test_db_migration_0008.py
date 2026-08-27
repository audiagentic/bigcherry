"""HI121 close-out step 6 (HI127): schema-7 to schema-8 winner_verification
migration. Follows the same minimal-fixture-DB convention as
test_db_migration.py's schema-2/3 migration test."""

from __future__ import annotations

import sqlite3
import unittest
from pathlib import Path

MIGRATION = Path(__file__).resolve().parents[3] / "sql" / "migrations" / "0008_winner_verification.sql"


class Migration0008Tests(unittest.TestCase):
    def _fixture_connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(":memory:")
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript(
            """
            CREATE TABLE schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            INSERT INTO schema_meta VALUES ('schema_version', '7');
            CREATE TABLE winner (winner_id INTEGER PRIMARY KEY);
            INSERT INTO winner (winner_id) VALUES (1);
            """
        )
        return connection

    def test_migration_adds_winner_verification_and_bumps_schema(self):
        connection = self._fixture_connection()
        connection.executescript(MIGRATION.read_text(encoding="utf-8"))
        self.assertEqual(
            connection.execute("SELECT value FROM schema_meta WHERE key='schema_version'").fetchone()[0],
            "8",
        )
        columns = {row[1] for row in connection.execute("PRAGMA table_info(winner_verification)")}
        self.assertTrue({"winner_id", "verification_profile", "verified_at"} <= columns)

    def test_migration_refuses_wrong_starting_schema_version(self):
        connection = sqlite3.connect(":memory:")
        connection.executescript(
            """
            CREATE TABLE schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            INSERT INTO schema_meta VALUES ('schema_version', '6');
            CREATE TABLE winner (winner_id INTEGER PRIMARY KEY);
            """
        )
        with self.assertRaises(sqlite3.IntegrityError):
            connection.executescript(MIGRATION.read_text(encoding="utf-8"))

    def test_deleting_winner_row_cascades_to_verification(self):
        connection = self._fixture_connection()
        connection.executescript(MIGRATION.read_text(encoding="utf-8"))
        connection.execute(
            "INSERT INTO winner_verification (winner_id, verification_profile) VALUES (1, 'hi121-strengthened-ingest-v1')"
        )
        connection.commit()
        connection.execute("DELETE FROM winner WHERE winner_id = 1")
        connection.commit()
        remaining = connection.execute("SELECT COUNT(*) FROM winner_verification").fetchone()[0]
        self.assertEqual(remaining, 0)

    def test_only_current_profile_string_is_accepted(self):
        connection = self._fixture_connection()
        connection.executescript(MIGRATION.read_text(encoding="utf-8"))
        with self.assertRaises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO winner_verification (winner_id, verification_profile) VALUES (1, 'some-other-profile')"
            )


if __name__ == "__main__":
    unittest.main()
