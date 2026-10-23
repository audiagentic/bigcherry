"""WSL06 legacy discovery is read-only, conservative and auditable."""
from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from pathlib import Path

from bigcherry.cli import workspace_cmds
from bigcherry.core.legacy_inventory import inventory_legacy


class LegacyInventoryTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        (self.root / "runs" / "ab" / "legacy-test").mkdir(parents=True)
        (self.root / "runs" / "ab" / "legacy-test" / "capture.bin").write_bytes(b"x" * 33)
        (self.root / "builds" / "recipe" / "build").mkdir(parents=True)
        (self.root / "sources" / "src").mkdir(parents=True)
        (self.root / "worktrees" / "old-checkout").mkdir(parents=True)
        (self.root / "worktrees" / "old-checkout" / ".git").write_text("gitdir: missing")
        (self.root / "legacy-cache").mkdir()

    def test_classifies_every_root_without_cleanup_candidates(self):
        report = inventory_legacy(self.root)
        self.assertTrue(report["read_only"])
        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(report["totals"]["entries"], 5)
        self.assertTrue(report["totals"]["incomplete"])
        self.assertTrue(all(e["action"] == "protected" for e in report["entries"]))
        self.assertEqual(
            {e["category"] for e in report["entries"]},
            {"run", "build", "source", "worktree", "unclassified-root"},
        )
        self.assertIn("legacy-cache", next(e["path"] for e in report["entries"]
                                           if e["category"] == "unclassified-root"))

    def test_complete_size_is_payload_not_reclamation_estimate(self):
        report = inventory_legacy(self.root, measure=True)
        run = next(e for e in report["entries"] if e["category"] == "run")
        self.assertTrue(run["size_complete"])
        self.assertEqual(run["size_bytes"], 33)
        self.assertFalse(report["totals"]["incomplete"])
        self.assertTrue(all(e["action"] == "protected" for e in report["entries"]))

    @unittest.skipIf(sys.platform == "win32", "symlink permissions vary")
    def test_symlink_is_reported_without_traversal(self):
        external = self.root.parent / (self.root.name + "-outside")
        external.mkdir()
        self.addCleanup(external.rmdir)
        (self.root / "unknown-link").symlink_to(external, target_is_directory=True)
        report = inventory_legacy(self.root, measure=True)
        link = next(e for e in report["entries"] if e["path"].endswith("unknown-link"))
        self.assertTrue(link["is_symlink"])
        self.assertIsNone(link["size_bytes"])
        self.assertEqual(link["action"], "protected")

    def test_cli_json_read_only_apply_refused(self):
        args = Namespace(work_root=str(self.root), legacy=True, builds=False,
                         no_size=True, json=True, apply=False,
                         min_age_days=7, keep_newest=3, verbose=False)
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(workspace_cmds.cmd_clean(args), 0)
        report = json.loads(out.getvalue())
        self.assertEqual(report["totals"]["protected"], 5)
        args.apply = True
        with redirect_stdout(io.StringIO()):
            self.assertEqual(workspace_cmds.cmd_clean(args), 2)
        self.assertTrue((self.root / "legacy-cache").exists())

    def test_read_only_refuses_root_symlink(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "link"
            path.symlink_to(self.root, target_is_directory=True)
            with self.assertRaises(ValueError):
                inventory_legacy(path)


if __name__ == "__main__":
    unittest.main()
