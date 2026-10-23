"""WSL03: destructive CLI admission is conservative across project worktrees."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from bigcherry.cli import workspace_cmds
from bigcherry.core import tree_activity


class CleanupActivityAdmission(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def lease_dir(self, namespace="different-checkout"):
        result = self.root / "tree-activity" / namespace / "leases"
        result.mkdir(parents=True, exist_ok=True)
        return result

    def test_idle_root_remains_unblocked(self):
        self.assertIsNone(tree_activity.cleanup_activity_blocker(self.root))
        self.assertIsNone(workspace_cmds._blocked(self.root))
        self.lease_dir()
        self.assertIsNone(tree_activity.cleanup_activity_blocker(self.root))

    def test_other_checkout_lease_blocks_mutation(self):
        (self.lease_dir() / "active.json").write_text('{"pid": 123}', encoding="utf-8")
        self.assertIn("different-checkout/active.json", tree_activity.cleanup_activity_blocker(self.root))
        self.assertIn("different-checkout/active.json", workspace_cmds._blocked(self.root))

    def test_bad_or_stale_lease_fails_closed(self):
        for payload in ("{broken", '{"pid": 999999999}', ""):
            with self.subTest(payload=payload):
                lease = self.lease_dir() / "claim.json"
                lease.write_text(payload, encoding="utf-8")
                self.assertIn("claim.json", workspace_cmds._blocked(self.root))

    def test_other_namespace_maintenance_lock_blocks_mutation(self):
        maintenance = self.root / "tree-activity" / "other-branch" / "maintenance.lock"
        maintenance.mkdir(parents=True)
        self.assertIn("other-branch", workspace_cmds._blocked(self.root))

    def test_unclassified_directory_entry_blocks_mutation(self):
        (self.lease_dir() / "broken-record").mkdir()
        self.assertIn("broken-record", workspace_cmds._blocked(self.root))

    @unittest.skipIf(sys.platform == "win32", "symlink privileges vary")
    def test_symlinked_activity_root_or_namespace_fails_closed(self):
        external = self.root / "external"
        external.mkdir()
        activity = self.root / "tree-activity"
        activity.symlink_to(external, target_is_directory=True)
        self.assertIn("symlink", workspace_cmds._blocked(self.root))
        activity.unlink()
        activity.mkdir()
        (activity / "another-namespace").symlink_to(external, target_is_directory=True)
        self.assertIn("unrecognised", workspace_cmds._blocked(self.root))


if __name__ == "__main__":
    unittest.main()
