"""ProjectContext path precedence and host-local defaults."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bigcherry.context import ProjectContext  # noqa: E402


class ProjectContextTests(unittest.TestCase):
    def test_explicit_roots_win_and_metadata_stays_outside_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            work = root / "work"
            ctx = ProjectContext.resolve(
                project_root=root,
                config_path=root / "cfg.toml",
                artifacts_root=root / "artifacts",
                work_root=work,
                upstream_repo=work / "objects.git",
            )
            self.assertEqual(ctx.project_root, root.resolve())
            self.assertEqual(ctx.upstream_repo, (work / "objects.git").resolve())
            self.assertNotEqual(ctx.work_root, ctx.project_root)
            self.assertNotEqual(ctx.work_root / "metadata", ctx.overlay_root)


if __name__ == "__main__":
    unittest.main()
