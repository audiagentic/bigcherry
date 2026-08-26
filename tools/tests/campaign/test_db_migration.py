"""Explicit schema-2 to schema-3 campaign identity migration."""

from __future__ import annotations

import sqlite3
import unittest
from pathlib import Path


class DatabaseMigrationTests(unittest.TestCase):
    def test_migration_adds_identity_columns_and_bumps_schema(self):
        connection = sqlite3.connect(":memory:")
        connection.executescript(
            """
            CREATE TABLE schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            INSERT INTO schema_meta VALUES ('schema_version', '2');
            CREATE TABLE build (build_id INTEGER PRIMARY KEY);
            CREATE TABLE observation (observation_id INTEGER PRIMARY KEY);
            CREATE TABLE measurement (measurement_id INTEGER PRIMARY KEY);
            CREATE TABLE winner (winner_id INTEGER PRIMARY KEY);
            CREATE TABLE replay_miss (miss_id INTEGER PRIMARY KEY);
            """
        )
        migration = Path(__file__).resolve().parents[3] / "sql" / "migrations" / "0003_campaign_identity.sql"
        connection.executescript(migration.read_text(encoding="utf-8"))
        self.assertEqual(
            connection.execute("SELECT value FROM schema_meta WHERE key='schema_version'").fetchone()[0],
            "3",
        )
        columns = {row[1] for row in connection.execute("PRAGMA table_info(build)")}
        self.assertTrue({"source_slice_id", "build_plan_id", "effective_build_id", "campaign_run_id", "workload_id"} <= columns)


if __name__ == "__main__":
    unittest.main()
