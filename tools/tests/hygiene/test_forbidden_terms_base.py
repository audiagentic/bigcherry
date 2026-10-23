"""Forbidden-term scan in diff mode: only files changed since a base revision are reported."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from bigcherry import forbidden_terms as ft


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True,
                          text=True).stdout.strip()


class ForbiddenTermsBaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        _git(self.root, "init", "-q")
        _git(self.root, "config", "user.email", "t@example.com")
        _git(self.root, "config", "user.name", "t")
        self._env = os.environ.pop("BIGCHERRY_FORBIDDEN_TERMS", None)
        os.environ["BIGCHERRY_FORBIDDEN_TERMS"] = "zeta-box"

    def tearDown(self) -> None:
        os.environ.pop("BIGCHERRY_FORBIDDEN_TERMS", None)
        if self._env is not None:
            os.environ["BIGCHERRY_FORBIDDEN_TERMS"] = self._env
        self._tmp.cleanup()

    def _commit(self, rel: str, text: str) -> str:
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        _git(self.root, "add", rel)
        _git(self.root, "commit", "-q", "-m", f"edit {rel}")
        return _git(self.root, "rev-parse", "HEAD")

    def test_only_files_changed_since_base_are_scanned(self) -> None:
        base = self._commit("docs/old.md", "already here: zeta-box\n")
        self._commit("docs/new.md", "clean text\n")
        self.assertEqual([], ft.findings(self.root, base=base))

    def test_a_changed_file_with_a_term_is_reported(self) -> None:
        base = self._commit("docs/a.md", "clean\n")
        self._commit("docs/a.md", "now on zeta-box\n")
        hits = ft.findings(self.root, base=base)
        self.assertEqual([("docs/a.md", 1, "machine name")], hits)


if __name__ == "__main__":
    unittest.main()
