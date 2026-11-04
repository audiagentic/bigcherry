"""BU01: end-to-end PR gate path selection using actual workflow patterns."""
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SELECTOR = ROOT / ".github/scripts/changed_paths.py"
WORKFLOW = ROOT / ".github/workflows/pr-ci.yml"


def patterns() -> str:
    source = WORKFLOW.read_text()
    section = source.split("      - id: radiance\n", 1)[1]
    match = re.search(r"<<'PATTERNS'\n(.*?)\n          PATTERNS", section, re.S)
    assert match, "radiance PR gate must retain the documented changed_paths.py pattern list"
    return "\n".join(line.strip() for line in match.group(1).splitlines()) + "\n"


def git(root: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(root), *args], check=True,
                            capture_output=True, text=True)
    return result.stdout.strip()


class ChangedPathsIntegration(unittest.TestCase):
    def setUp(self):
        work = ROOT / "work" / "bu01-offline-tests"
        work.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=work)
        self.repo = Path(self.temp.name)
        git(self.repo, "init", "-q")
        git(self.repo, "config", "user.name", "BU01 Test")
        git(self.repo, "config", "user.email", "bu01@example.invalid")
        (self.repo / "init").write_text("test")
        git(self.repo, "add", "-A")
        git(self.repo, "commit", "-q", "-m", "base")
        self.base = git(self.repo, "rev-parse", "HEAD")

    def tearDown(self):
        self.temp.cleanup()

    def check(self, files: dict[str, str]) -> str:
        for name, content in files.items():
            dest = self.repo / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content)
        git(self.repo, "add", "-A")
        git(self.repo, "commit", "-q", "-m", "updated")
        head = git(self.repo, "rev-parse", "HEAD")
        p = subprocess.run([sys.executable, str(SELECTOR), self.base, head],
                           cwd=self.repo, input=patterns(), capture_output=True, text=True,
                           check=True)
        return p.stdout.strip()

    def test_markdown_only_skips(self):
        self.assertEqual(self.check({
            "engines/radiance/components/moe-remote/README.md": "docs",
            "engines/radiance/docs/planning/BU01.md": "plan",
        }), "run=false")

    def test_kernel_edits_still_build(self):
        self.assertEqual(self.check({
            "engines/radiance/components/moe-remote/mailbox_inverse_kernel.hip": "kernel"
        }), "run=true")

    def test_mixed_docs_and_kernel_still_build(self):
        self.assertEqual(self.check({
            "engines/radiance/README.md": "docs",
            "engines/radiance/components/libr3/CMakeLists.txt": "cmake"
        }), "run=true")

    def test_component_test_and_patches_build(self):
        self.assertEqual(self.check({"engines/radiance/tests/check.py": "test"}), "run=true")

    def test_diff_failure_runs_checks(self):
        p = subprocess.run([sys.executable, str(SELECTOR), "not_a_sha", "also_bad"],
                           cwd=self.repo, input=patterns(), capture_output=True, text=True,
                           check=True)
        self.assertEqual(p.stdout.strip(), "run=true")

    def test_ci_pass_accepts_skipped_area(self):
        text = WORKFLOW.read_text()
        self.assertIn('if: always()', text)
        self.assertIn('("success", "skipped")', text)


if __name__ == "__main__":
    unittest.main()
