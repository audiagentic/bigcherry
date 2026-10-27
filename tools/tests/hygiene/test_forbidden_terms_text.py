"""Forbidden-term scan of free text: commit messages and pull request titles/descriptions."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from bigcherry import forbidden_terms as ft


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


class ForbiddenTermsTextTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        _git(self.root, "init", "-q")
        _git(self.root, "config", "user.email", "t@example.com")
        _git(self.root, "config", "user.name", "t")
        self._env = os.environ.get("BIGCHERRY_FORBIDDEN_TERMS")
        os.environ["BIGCHERRY_FORBIDDEN_TERMS"] = "zeta-box"

    def tearDown(self) -> None:
        if self._env is None:
            os.environ.pop("BIGCHERRY_FORBIDDEN_TERMS", None)
        else:
            os.environ["BIGCHERRY_FORBIDDEN_TERMS"] = self._env
        self._tmp.cleanup()

    def _commit(self, message: str) -> None:
        (self.root / "f.txt").write_text(message, encoding="utf-8")
        _git(self.root, "add", "f.txt")
        _git(self.root, "commit", "-q", "-m", message)

    def test_pr_text_with_a_machine_name_is_found(self) -> None:
        hits = ft.text_findings(self.root, "pr", "title\n\nResults on Zeta-Box, 60 s holds\n")
        self.assertEqual(hits, [("pr", 3, "machine name")])

    def test_clean_text_has_no_findings(self) -> None:
        self.assertEqual(ft.text_findings(self.root, "pr", "Results on the bench host\n"), [])

    def test_commit_messages_since_base_are_scanned(self) -> None:
        self._commit("base")
        base = subprocess.run(["git", "-C", str(self.root), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
        self._commit("clean change")
        self._commit("measured on zeta-box")
        hits = ft.message_findings(self.root, base)
        self.assertEqual(len(hits), 1)
        self.assertTrue(hits[0][0].startswith("commit ") and hits[0][2] == "machine name")


if __name__ == "__main__":
    unittest.main()
