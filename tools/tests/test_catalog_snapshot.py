"""RE39: CatalogSnapshot -- one immutable read of both patch registries."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bigcherry import patch_catalog  # noqa: E402


class CatalogSnapshotTests(unittest.TestCase):
    def test_build_snapshot_on_the_real_catalog(self):
        snapshot = patch_catalog.build_snapshot()
        self.assertGreater(len(snapshot.modules), 0)
        self.assertEqual(len(snapshot.metadata), len(snapshot.modules))
        self.assertTrue(snapshot.digest)

    def test_by_id_keys_match_metadata_keys(self):
        snapshot = patch_catalog.build_snapshot()
        self.assertEqual(set(snapshot.by_id), set(snapshot.metadata))

    def test_entry_for_returns_none_for_unknown_id(self):
        snapshot = patch_catalog.build_snapshot()
        self.assertIsNone(snapshot.entry_for("not-a-real-patch-id"))

    def test_entry_for_returns_the_real_catalog_entry(self):
        snapshot = patch_catalog.build_snapshot()
        entry = snapshot.entry_for("0100_cmake_options")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.kind, "framework")

    def test_digest_is_stable_across_two_builds(self):
        first = patch_catalog.build_snapshot()
        second = patch_catalog.build_snapshot()
        self.assertEqual(first.digest, second.digest)

    def test_digest_changes_if_a_module_is_removed_from_the_input_set(self):
        # Simulate a smaller catalog by pointing at a subset via a temp dir
        # would require real files; instead prove sensitivity structurally:
        # two snapshots built from genuinely different metadata differ.
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as handle:
            handle.write("""
version = 1
[[patch]]
id = "0100_cmake_options"
kind = "framework"
origin = "local"
backend = "hip"
state = "validated"
""")
            small_catalog = Path(handle.name)
        try:
            full = patch_catalog.build_snapshot()
            partial = patch_catalog.build_snapshot(catalog_path=small_catalog)
            self.assertNotEqual(full.digest, partial.digest)
        finally:
            small_catalog.unlink()

    def test_snapshot_modules_carry_real_requires_conflicts(self):
        # RE40's backfill should be visible through the snapshot's modules,
        # not just via a direct patchset.catalog() call.
        snapshot = patch_catalog.build_snapshot()
        module = snapshot.by_id["1217_rd44_graph_opt_default_rdna35"]
        self.assertEqual(
            module.requires,
            ("1215_rd394041_amd_stream_moe_overlap", "1216_rd43_concurrent_join_fusion_guard"),
        )


if __name__ == "__main__":
    unittest.main()
