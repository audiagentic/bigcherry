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

    def test_a_long_base64_line_scans_in_linear_time(self) -> None:
        import time

        line = "QUJD" * 600_000
        started = time.monotonic()
        self.assertEqual(ft.text_findings(self.root, "pr", line + "\n"), [])
        self.assertLess(time.monotonic() - started, 5.0)
        self.assertEqual(ft.text_findings(self.root, "pr", "mail " + "a.b+c" + "@corp.io\n"),
                         [("pr", 1, "email address")])

    def test_pr_text_with_a_machine_name_is_found(self) -> None:
        hits = ft.text_findings(self.root, "pr", "title\n\nResults on Zeta-Box, 60 s holds\n")
        self.assertEqual(hits, [("pr", 3, "machine name")])

    def test_clean_text_has_no_findings(self) -> None:
        self.assertEqual(ft.text_findings(self.root, "pr", "Results on the bench host\n"), [])

    def test_attribution_trailer_is_allowed(self) -> None:
        (self.root / "config").mkdir()
        (self.root / "config" / "forbidden-terms-allow.txt").write_text("noreply@anthropic.com\n", encoding="utf-8")
        msg = "fix: thing\n\nCo-Authored-By: Claude <noreply@anthropic.com>\n"
        self.assertEqual(ft.text_findings(self.root, "commit", msg), [])

    def test_commit_messages_since_base_are_scanned(self) -> None:
        self._commit("base")
        base = subprocess.run(["git", "-C", str(self.root), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
        self._commit("clean change")
        self._commit("measured on zeta-box")
        hits = ft.message_findings(self.root, base)
        self.assertEqual(len(hits), 1)
        self.assertTrue(hits[0][0].startswith("commit ") and hits[0][2] == "machine name")

    def test_only_git_commit_trailer_may_hide_github_noreply(self):
        self._commit("base")
        base = subprocess.run(["git", "-C", str(self.root), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
        angle = lambda value: chr(60) + value + chr(62)
        gh_domain = "@users" + ".noreply.github.com"
        ordinary_domain = "@ordinary" + ".org"
        self._commit("clean change\n\nCo-authored-by: Contributor " + angle("1234+user" + gh_domain))
        self.assertEqual([], ft.message_findings(self.root, base))
        self._commit("other author\n\nCo-authored-by: Person " + angle("person" + ordinary_domain))
        self.assertEqual(["email address"], [x[2] for x in ft.message_findings(self.root, base)])
        self._commit("embedded identity person" + gh_domain)
        self.assertEqual(2, len(ft.message_findings(self.root, base)))
        self._commit("identity\n\nCo-authored-by: zeta-box " + angle("1234+handle" + gh_domain))
        self.assertEqual(["email address", "email address", "machine name"],
                         sorted(x[2] for x in ft.message_findings(self.root, base)))
        self.assertTrue(ft.text_findings(self.root, "pr", "user" + gh_domain))


if __name__ == "__main__":
    unittest.main()
