"""Forbidden-term scan: staged and tracked files, allowlist, placeholder domains, term sources."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from bigcherry import forbidden_terms as ft


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


class ForbiddenTermsTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        _git(self.root, "init", "-q")
        _git(self.root, "config", "user.email", "t@example.com")
        _git(self.root, "config", "user.name", "t")
        (self.root / "config").mkdir()
        self._env = os.environ.pop("BIGCHERRY_FORBIDDEN_TERMS", None)

    def tearDown(self) -> None:
        if self._env is not None:
            os.environ["BIGCHERRY_FORBIDDEN_TERMS"] = self._env
        self._tmp.cleanup()

    def _stage(self, rel: str, text: str) -> None:
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        _git(self.root, "add", rel)

    def test_term_from_environment_is_detected_in_staged_content(self) -> None:
        os.environ["BIGCHERRY_FORBIDDEN_TERMS"] = "zeta-box"
        self._stage("tools/x.py", "# runs on zeta-box\n")
        hits = ft.findings(self.root, staged=True)
        self.assertEqual([("tools/x.py", 1, "machine name")], hits)

    def test_term_from_local_config_is_detected(self) -> None:
        (self.root / "config" / "environment.local.toml").write_text(
            '[forbidden]\nterms = ["zeta-box"]\n', encoding="utf-8")
        self._stage("docs/a.md", "Ran on zeta-box yesterday.\n")
        self.assertTrue(any(label == "machine name" for _, _, label in ft.findings(self.root, staged=True)))

    def test_generic_patterns_are_reported_by_label(self) -> None:
        # Built from parts so this file holds no literal host path or address of its own.
        mount_path = "/" + "mnt" + "/data/work"
        private_ip = ".".join(["10", "1", "2", "3"])
        self._stage("config/a.toml", f'root = "{mount_path}"\nhost = "{private_ip}"\n')
        labels = sorted(label for _, _, label in ft.findings(self.root, staged=True))
        self.assertEqual(["absolute mount path", "private IPv4 address"], labels)

    def test_allowlisted_phrase_containing_a_term_is_not_reported(self) -> None:
        os.environ["BIGCHERRY_FORBIDDEN_TERMS"] = "gpu_host"
        (self.root / "config" / "forbidden-terms-allow.txt").write_text(
            "integrated_gpu_host_buffer\n", encoding="utf-8")
        self._stage("engines/p/patch.toml", 'id = "1209_integrated_gpu_host_buffer_backout"\n')
        self.assertEqual([], ft.findings(self.root, staged=True))

    def test_exempt_file_is_skipped_and_others_still_scanned(self) -> None:
        os.environ["BIGCHERRY_FORBIDDEN_TERMS"] = "zeta-box"
        (self.root / "config" / "forbidden-terms-exempt.txt").write_text(
            "# evidence-bound\nengines/e/patches/p/patch.py\n", encoding="utf-8")
        self._stage("engines/e/patches/p/patch.py", "# validated on zeta-box\n")
        self._stage("engines/e/patches/q/patch.py", "# validated on zeta-box\n")
        self.assertEqual([("engines/e/patches/q/patch.py", 1, "machine name")], ft.findings(self.root, staged=True))

    def test_reserved_example_domains_are_placeholders(self) -> None:
        self._stage("tests/t.py", "address = 'user@example.com'\nother = 'a@b.test'\n")
        self.assertEqual([], ft.findings(self.root, staged=True))

    def test_version_numbers_are_not_addresses(self) -> None:
        self._stage("docs/v.md", "built with 7.2.4.0 and 10.2.1\n")
        self.assertEqual([], ft.findings(self.root, staged=True))

    def test_staged_scan_ignores_unstaged_edits(self) -> None:
        os.environ["BIGCHERRY_FORBIDDEN_TERMS"] = "zeta-box"
        self._stage("tools/ok.py", "x = 1\n")
        (self.root / "tools" / "ok.py").write_text("# zeta-box\n", encoding="utf-8")
        self.assertEqual([], ft.findings(self.root, staged=True))


if __name__ == "__main__":
    unittest.main()
