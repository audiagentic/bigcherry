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
PUSH_WORKFLOW = ROOT / ".github/workflows/radiance-build.yml"


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

    def test_patches_build(self):
        self.assertEqual(self.check({"engines/radiance/patches/rad0001_x/patch.py": "patch"}), "run=true")

    def test_patch_tests_and_runtime_layout_data_skip(self):
        # read by the offline patch and core tests, never by the lane build or the component tests
        self.assertEqual(self.check({
            "engines/radiance/tests/check.py": "test",
            "engines/radiance/tests/fixtures/abc/core/config.cpp": "fixture",
            "engines/radiance/layouts.toml": "layouts",
            "engines/radiance/archs.toml": "archs",
            "engines/radiance/model_constant_guard.toml": "guard",
        }), "run=false")

    def test_build_tooling_and_config_build(self):
        # the build command, source materialisation, patch application and evidence gates, and the configuration
        # the lane build reads
        for name in ("tools/bigcherry/__main__.py", "tools/bigcherry/cli/build.py",
                     "tools/bigcherry/source/workspace.py", "tools/bigcherry/patch/apply.py",
                     "tools/bigcherry/patch/gates.py", "tools/bigcherry/release/pin_status.py",
                     "config/recipes.toml", "config/release.toml", "tools/lab/radiance/ci_ccache.py"):
            self.assertEqual(self.check({name: name}), "run=true", name)

    def test_push_paths_match_the_pr_check(self):
        source = PUSH_WORKFLOW.read_text()
        block = re.search(r"\n    paths:\n((?:      - '[^']+'\n)+)", source)
        assert block, "radiance-build.yml must keep its push paths list"
        push = re.findall(r"      - '([^']+)'", block.group(1))
        pr = [line for line in patterns().splitlines() if line and not line.startswith("#")]
        # the PR check also reruns when its own workflow or the matcher changes
        self.assertEqual(sorted(push), sorted(set(pr) - {".github/workflows/pr-ci.yml",
                                                         ".github/scripts/changed_paths.py"}))

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
