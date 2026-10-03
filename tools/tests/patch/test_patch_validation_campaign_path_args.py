"""PA43 follow-up: validation_campaign path arguments are absolute and the
--worktree-root default is project-local.

Source worktrees are created with a ``worktree add <path>`` issued against
the vendor llama.cpp repository, so a relative --worktree-root (e.g.
``work/worktrees/pa40``) used to resolve against the wrong repository and
fail with exit 128. --workdir/--build-root/--worktree-root are therefore
resolved to absolute paths at argument-parse time.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.core.context import ProjectContext  # noqa: E402
from bigcherry.patch import validation_campaign as vc  # noqa: E402


class PathArgumentTests(unittest.TestCase):
    _REQUIRED = [
        "--patch", "0100_cmake_options", "--model", "model.gguf",
        "--hip-path", "rocm", "--amdgpu-targets", "gfx1100",
        "--manifest", "manifest.json",
    ]

    def _parsed(self, extra: list[str]):
        captured = {}

        def fake_run(args):  # noqa: ANN001
            captured["args"] = args
            return 0

        with mock.patch.object(vc, "run", side_effect=fake_run):
            self.assertEqual(vc.main([*self._REQUIRED, *extra]), 0)
        return captured["args"]

    def test_relative_path_arguments_are_resolved_to_absolute(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                args = self._parsed(
                    [
                        "--workdir", "work/run",
                        "--build-root", "work/builds",
                        "--worktree-root", "work/worktrees/pa40",
                    ]
                )
            finally:
                os.chdir(cwd)
        base = Path(tmp).resolve()
        self.assertEqual(args.workdir, base / "work" / "run")
        self.assertEqual(args.build_root, base / "work" / "builds")
        self.assertEqual(args.worktree_root, base / "work" / "worktrees" / "pa40")
        for value in (args.workdir, args.build_root, args.worktree_root):
            self.assertTrue(value.is_absolute())

    def test_worktree_root_defaults_under_the_project_work_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = self._parsed(["--workdir", str(Path(tmp) / "run")])
        self.assertEqual(
            args.worktree_root, ProjectContext.resolve().work_root / "worktrees"
        )


if __name__ == "__main__":
    unittest.main()
