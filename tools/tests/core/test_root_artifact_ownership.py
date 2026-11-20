"""Contracts for engine-owned dispositions, platform SQL and isolated test scratch."""

from __future__ import annotations

import sqlite3
import unittest

from bigcherry.core import paths
from bigcherry.patch import disposition, offline_check


class RootArtifactOwnershipTests(unittest.TestCase):
    def test_dispositions_are_tracked_with_llamacpp(self) -> None:
        self.assertEqual(paths.PATCHES.parent / "dispositions", paths.DISPOSITIONS)
        records = disposition.list_dispositions(paths.DISPOSITIONS)
        self.assertEqual(6, len(records))
        self.assertNotEqual({}, records)
        self.assertTrue(all(item.patch_id == key for key, item in records.items()))
        self.assertFalse((paths.REPO_ROOT / "dispositions").exists())

    def test_changed_disposition_path_triggers_patch_validation(self) -> None:
        rel = "engines/llamacpp/dispositions/1267_rd07_q6k_mmq_scale_fold.json"
        self.assertIn("1267_rd07_q6k_mmq_scale_fold", offline_check._patch_ids([rel]))
        self.assertNotIn(
            "1267_rd07_q6k_mmq_scale_fold",
            offline_check._patch_ids(["dispositions/1267_rd07_q6k_mmq_scale_fold.json"]),
        )

    def test_shared_schema_bootstraps_a_real_sqlite_database(self) -> None:
        self.assertEqual(
            paths.REPO_ROOT / "tools/bigcherry/tuning/sql",
            paths.SQL,
        )
        self.assertFalse((paths.REPO_ROOT / "sql").exists())
        sql = (paths.SQL / "dispatch-db.sql").read_text(encoding="utf-8")
        with sqlite3.connect(":memory:") as conn:
            conn.executescript(sql)
            self.assertIsNotNone(
                conn.execute("SELECT value FROM schema_meta WHERE key = 'schema_version'").fetchone()
            )
            self.assertIsNotNone(
                conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='winner'"
                ).fetchone()
            )

    def test_host_experiment_is_cataloged_and_source_fixture_is_absent(self) -> None:
        experiment = paths.REPO_ROOT / "experiments/host-kernel-gfxoff-7.0.13"
        self.assertTrue((experiment / "README.md").is_file())
        self.assertTrue((experiment / "series").is_file())
        self.assertTrue((experiment / "patches").is_dir())
        self.assertFalse((paths.REPO_ROOT / "host").exists())
        self.assertFalse((paths.REPO_ROOT / "bc-owner-map-fixture").exists())


if __name__ == "__main__":
    unittest.main()
