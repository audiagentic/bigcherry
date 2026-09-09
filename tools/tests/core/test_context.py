"""ProjectContext path precedence and host-local defaults."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.core.context import ProjectContext  # noqa: E402


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
                upstream_repo=root / "upstream.git",
            )
            self.assertEqual(ctx.project_root, root.resolve())
            self.assertEqual(ctx.upstream_repo, (root / "upstream.git").resolve())
            self.assertNotEqual(ctx.work_root, ctx.project_root)
            self.assertNotEqual(ctx.work_root / "metadata", ctx.overlay_root)

    def test_equal_roots_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "must be disjoint"):
                ProjectContext.resolve(work_root=root, upstream_repo=root)

    def test_nested_roots_are_rejected_in_either_direction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for work, upstream in (
                (root / "work", root / "work" / "upstream.git"),
                (root / "work" / "nested", root / "work"),
            ):
                with self.subTest(work=work, upstream=upstream):
                    with self.assertRaisesRegex(ValueError, "must be disjoint"):
                        ProjectContext.resolve(work_root=work, upstream_repo=upstream)

    def test_sibling_roots_remain_valid(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ctx = ProjectContext.resolve(
                work_root=root / "work",
                upstream_repo=root / "work-old" / "upstream.git",
            )
            self.assertEqual(ctx.work_root, (root / "work").resolve())


if __name__ == "__main__":
    unittest.main()
