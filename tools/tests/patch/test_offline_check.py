"""Regression tests for the changed-patch offline checker."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from bigcherry.patch import offline_check


class OfflineCheckLifecycleTests(unittest.TestCase):
    def test_patch_state_reads_lifecycle(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "patch.toml"
            path.write_text('state = "rejected"\n', encoding="utf-8")
            self.assertEqual(offline_check._patch_state(path), "rejected")

    def test_retired_changed_patch_is_not_focal_rebased(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            patch_id = "1268_prbe52_adaptive_mtp_wiring"
            package = root / "patches" / patch_id
            package.mkdir(parents=True)
            (package / "patch.toml").write_text('state = "rejected"\n', encoding="utf-8")
            test_path = root / "tools" / "tests" / "patch" / f"test_{patch_id}.py"
            test_path.parent.mkdir(parents=True)
            test_path.write_text("", encoding="utf-8")

            calls = []

            def fake_run(name, command, checks):
                calls.append((name, tuple(command)))
                checks.append({"name": name, "command": list(command), "returncode": 0})
                return 0

            with (
                mock.patch.object(offline_check, "_REPO", root),
                mock.patch.object(offline_check, "_resolve", side_effect=["base-sha", "head-sha"]),
                mock.patch.object(
                    offline_check,
                    "_changed_files",
                    return_value=(f"patches/{patch_id}/patch.toml",),
                ),
                mock.patch.object(offline_check, "_run", side_effect=fake_run),
            ):
                self.assertEqual(
                    offline_check.main(["--base", "base", "--head", "head", "--source", "bigcherry"]),
                    0,
                )

            commands = [command for _, command in calls]
            self.assertFalse(
                any("--focal-overlay" in command and patch_id in command for command in commands)
            )
            self.assertTrue(
                any(
                    "patch-rebase-check" in command
                    and "--source" in command
                    and "--focal-overlay" not in command
                    for command in commands
                )
            )


if __name__ == "__main__":
    unittest.main()
