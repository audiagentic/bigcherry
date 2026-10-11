"""pin-release: version pinning, PR publication, resume behavior and dry-run reporting."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bigcherry.release import notes as _notes  # noqa: E402
from bigcherry.release import pin_release as _pr  # noqa: E402

_CONFIG = _notes.ReleaseConfig(
    tag_prefix="bc-", notes_dir="docs/releases/notes", ledger="ledger.ndjson", release_records="releases",
    recipes="config/recipes.toml", build_patch_sets=(), fallback_prefixes=(), ledger_sections=(),
)

_RELEASE_CONFIG = """\
{
  "packages": {
    "engines/llamacpp": {
      "release-type": "simple",
      "component": "bc-llamacpp",
      "changelog-path": "CHANGELOG.md"
    },
    ".": {
      "release-type": "simple",
      "component": "bc-platform",
      "changelog-path": "docs/releases/PLATFORM_CHANGELOG.md"
    }
  }
}
"""


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True).stdout.strip()


class PinReleaseTests(unittest.TestCase):
    def _repo(self, td: str) -> Path:
        root = Path(td)
        _git(root, "init", "-q", "-b", "work")
        _git(root, "config", "user.email", "t@example.invalid")
        _git(root, "config", "user.name", "t")
        (root / "releases").mkdir()
        (root / "releases" / "b11474.json").write_text(json.dumps({"release_tag": "b11474", "notes": ""}), encoding="utf-8")
        (root / "releases" / "pin-transition.json").write_text("{}", encoding="utf-8")
        (root / "release-please-config.json").write_text(_RELEASE_CONFIG, encoding="utf-8")
        (root / ".release-please-manifest.json").write_text(
            json.dumps({"engines/llamacpp": "11402.0.0", ".": "1.1.0"}, indent=2) + "\n", encoding="utf-8")
        _git(root, "add", "-A")
        _git(root, "commit", "-q", "-m", "seed")
        return root

    def _plan(self, root: Path) -> _pr.Plan:
        with mock.patch.object(_notes, "load_config", return_value=_CONFIG):
            return _pr.make_plan(root, "b11474")

    def test_plan_names(self):
        with tempfile.TemporaryDirectory() as td:
            plan = self._plan(self._repo(td))
            self.assertEqual((plan.build, plan.release_tag, plan.version, plan.branch),
                             (11474, "bc-11474.0.0", "11474.0.0", "work"))

    def test_plan_rejects_other_tags_and_main(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._repo(td)
            with mock.patch.object(_notes, "load_config", return_value=_CONFIG):
                with self.assertRaises(_pr.PinReleaseError):
                    _pr.make_plan(root, "v1.2.3")
                _git(root, "checkout", "-q", "-b", "main")
                with self.assertRaises(_pr.PinReleaseError):
                    _pr.make_plan(root, "b11474")

    def test_record_needs_evidence_then_records_once(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._repo(td)
            plan = self._plan(root)
            with self.assertRaises(_pr.PinReleaseError):
                _pr.phase_record(plan, "VERDICT consistent", "")
            self.assertTrue((root / "releases" / "pin-transition.json").exists())
            self.assertTrue(_pr.phase_record(plan, "VERDICT consistent", "build b-x: smoke identical"))
            record = json.loads((root / "releases" / "b11474.json").read_text(encoding="utf-8"))
            self.assertIn("release 11474.0.0", record["notes"])
            self.assertIn("VERDICT consistent", record["notes"])
            self.assertIn("build b-x: smoke identical", record["notes"])
            self.assertFalse((root / "releases" / "pin-transition.json").exists())
            self.assertEqual(_git(root, "status", "--porcelain"), "")
            self.assertIn("bc-11474.0.0 bump complete", _git(root, "log", "-1", "--format=%s"))
            head = _git(root, "rev-parse", "HEAD")
            self.assertFalse(_pr.phase_record(plan, "VERDICT consistent", ""))
            self.assertEqual(head, _git(root, "rev-parse", "HEAD"))

    def test_notes_sets_release_as_on_llamacpp_only_and_commits_no_footer(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._repo(td)
            plan = self._plan(root)
            original_config = (root / "release-please-config.json").read_text(encoding="utf-8")

            def fake_write(repo_root, config, llama_tag, ref="HEAD", version=None):
                out = Path(repo_root) / "docs" / "releases" / "notes" / f"bc-{version}.md"
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text("# notes\n", encoding="utf-8")
                return out

            with mock.patch.object(_notes, "load_config", return_value=_CONFIG), \
                    mock.patch.object(_notes, "write_notes", side_effect=fake_write):
                self.assertTrue(_pr.phase_notes(plan))
            text = (root / "release-please-config.json").read_text(encoding="utf-8")
            config = json.loads(text)
            self.assertEqual(config["packages"]["engines/llamacpp"]["release-as"], "11474.0.0")
            self.assertNotIn("release-as", config["packages"]["."])
            self.assertEqual(text.replace('      "release-as": "11474.0.0",\n', ""), original_config)
            self.assertNotIn("Release-As", _git(root, "log", "-1", "--format=%B"))
            head = _git(root, "rev-parse", "HEAD")
            self.assertFalse(_pr.phase_notes(plan))
            self.assertEqual(head, _git(root, "rev-parse", "HEAD"))

    def test_notes_refuses_conflicting_release_as_before_writing(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._repo(td)
            plan = self._plan(root)
            path = root / "release-please-config.json"
            path.write_text(path.read_text(encoding="utf-8").replace(
                '    "engines/llamacpp": {', '    "engines/llamacpp": {\n      "release-as": "999.0.0",'),
                encoding="utf-8")
            with mock.patch.object(_notes, "write_notes") as write:
                with self.assertRaisesRegex(_pr.PinReleaseError, "refusing to replace"):
                    _pr.phase_notes(plan)
            write.assert_not_called()

    def test_unpin_removes_release_as_and_is_noop_when_absent(self):
        pinned = _RELEASE_CONFIG.replace(
            '    "engines/llamacpp": {', '    "engines/llamacpp": {\n      "release-as": "11474.0.0",')
        unpinned, changed = _pr._set_release_as_text(pinned, None, "unpin", expected="11474.0.0")
        self.assertTrue(changed)
        self.assertEqual(unpinned, _RELEASE_CONFIG)
        again, changed = _pr._set_release_as_text(unpinned, None, "unpin", expected="11474.0.0")
        self.assertFalse(changed)
        self.assertEqual(again, unpinned)

    def test_plan_resumes_the_tagged_release_for_post_tag_phases(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._repo(td)
            (root / "resume.txt").write_text("resume\n", encoding="utf-8")
            _git(root, "add", "resume.txt")
            _git(root, "commit", "-q", "-m", "chore: release notes bc-11474.0.0")
            _git(root, "tag", "bc-11474.0.0")
            with mock.patch.object(_notes, "load_config", return_value=_CONFIG):
                plan = _pr.make_plan(root, "b11474")
            self.assertEqual(plan.version, "11474.0.0")

    def test_versions_on_one_pin(self):
        self.assertEqual(_pr.next_version([], 11474, None), "11474.0.0")
        self.assertEqual(_pr.next_version(["11402.0.0", "b11474"], 11474, None), "11474.0.0")
        self.assertEqual(_pr.next_version(["11474.0.0"], 11474, "minor"), "11474.1.0")
        self.assertEqual(_pr.next_version(["11474.0.0", "11474.1.0"], 11474, "patch"), "11474.1.1")
        self.assertEqual(_pr.next_version(["11474.0.0", "11474.1.1"], 11474, "minor"), "11474.2.0")
        with self.assertRaises(_pr.PinReleaseError):
            _pr.next_version(["11474.0.0"], 11474, None)
        with self.assertRaises(_pr.PinReleaseError):
            _pr.next_version([], 11474, "minor")
        self.assertEqual(_notes.release_version("b11474"), "11474.0.0")
        self.assertEqual(_notes.release_version("b11474", "11474.2.1"), "11474.2.1")
        with self.assertRaises(_notes.ReleaseNotesError):
            _notes.release_version("b11474", "11402.1.0")

    def test_main_pushes_only_work_branch_and_opens_or_reuses_pr(self):
        plan = SimpleNamespace(repo_root=Path("/repo"), remote="origin", main="main", branch="work",
                               llama_tag="b11474", release_tag="bc-11474.0.0")
        git_calls = []

        def fake_git(root, *args, check=True):
            git_calls.append(args)
            if args[:2] == ("status", "--porcelain"):
                return ""
            if args[:2] == ("rev-parse", "HEAD"):
                return "head"
            return ""

        ancestry = [
            SimpleNamespace(returncode=1, stdout="", stderr=""),
            SimpleNamespace(returncode=0, stdout="", stderr=""),
        ]

        def fake_run(root, *cmd, check=True, timeout=600):
            if cmd[:3] == ("git", "merge-base", "--is-ancestor"):
                return ancestry.pop(0)
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        pr = {"number": 7, "url": "u/7", "title": "build: release"}
        with mock.patch.object(_pr, "_git", side_effect=fake_git), \
                mock.patch.object(_pr, "_run", side_effect=fake_run), \
                mock.patch.object(_pr, "_ensure_pr", return_value=pr) as ensure, \
                mock.patch.object(_pr, "_wait_and_merge_pr") as merge:
            self.assertTrue(_pr.phase_main(plan))
        self.assertIn(("push", "-q", "origin", "HEAD:refs/heads/work"), git_calls)
        self.assertFalse(any("refs/heads/main" in arg for call in git_calls for arg in call))
        ensure.assert_called_once()
        merge.assert_called_once()

    def test_main_failing_check_stops_before_merge(self):
        plan = SimpleNamespace(repo_root=Path("/repo"))
        pr = {"number": 7, "url": "u/7"}
        failed = SimpleNamespace(returncode=1, stdout="failed check", stderr="")
        with mock.patch.object(_pr, "_run", return_value=failed) as run:
            with self.assertRaisesRegex(_pr.PinReleaseError, "checks failed"):
                _pr._wait_and_merge_pr(plan, pr, "main")
        self.assertFalse(any(call.args[1:4] == ("gh", "pr", "merge") for call in run.call_args_list))

    def test_ensure_pr_reuses_open_pr_from_same_branch(self):
        plan = SimpleNamespace(repo_root=Path("/repo"), main="main")
        existing = {"number": 9, "headRefName": "work", "title": "build: existing", "url": "u/9"}
        with mock.patch.object(_pr, "_pr_for_head", return_value=existing), mock.patch.object(_pr, "_run") as run:
            self.assertIs(_pr._ensure_pr(plan, "work", "build: new", "main", "body"), existing)
        run.assert_not_called()

    def test_unpin_phase_is_noop_when_main_is_already_clear(self):
        plan = SimpleNamespace(repo_root=Path("/repo"), remote="origin", main="main", version="11474.0.0",
                               release_tag="bc-11474.0.0")
        calls = []

        def fake_git(root, *args, check=True):
            calls.append(args)
            if args[:2] == ("show", "origin/main:release-please-config.json"):
                return _RELEASE_CONFIG
            return ""

        with mock.patch.object(_pr, "_tag_exists", return_value=True), \
                mock.patch.object(_pr, "_git", side_effect=fake_git), \
                mock.patch.object(_pr, "_ensure_pr") as ensure:
            self.assertFalse(_pr.phase_unpin(plan))
        ensure.assert_not_called()

    def test_phase_order_has_unpin_before_sync(self):
        self.assertEqual(_pr.PHASES, ("gate", "record", "notes", "main", "release", "unpin", "sync"))

    def test_full_dry_run_uses_only_read_git_calls_when_release_is_done(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._repo(td)
            plan = self._plan(root)
            calls = []
            real_git = _pr._git

            def recording_git(repo_root, *args, check=True):
                calls.append(args)
                return real_git(repo_root, *args, check=check)

            lines = []
            with mock.patch.object(_pr, "_tag_exists", return_value=True), \
                    mock.patch.object(_pr, "_git", side_effect=recording_git), \
                    mock.patch.object(_pr, "_pr_for_head") as pr_for_head, \
                    mock.patch.object(_pr, "_release_pr") as release_pr:
                _pr._dry_run(plan, _pr.PHASES.index("sync"), lines.append)
            writes = {"push", "commit", "add", "rm", "merge", "worktree"}
            self.assertFalse(any(call and call[0] in writes for call in calls))
            pr_for_head.assert_not_called()
            release_pr.assert_not_called()
            self.assertTrue(any("release  already done" in line for line in lines))
            self.assertTrue(any("manifest engines/llamacpp: 11402.0.0" in line for line in lines))

    def test_dry_run_lists_concrete_early_phases_and_changes_nothing(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._repo(td)
            head = _git(root, "rev-parse", "HEAD")
            config = (root / "release-please-config.json").read_text(encoding="utf-8")
            lines = []
            with mock.patch.object(_notes, "load_config", return_value=_CONFIG):
                self.assertEqual(_pr.run(root, "b11474", "", through="notes", dry_run=True, log=lines.append), 0)
            self.assertTrue(any("pin-status --complete --all-remotes" in line for line in lines))
            self.assertTrue(any("release-as=11474.0.0" in line for line in lines))
            self.assertEqual(head, _git(root, "rev-parse", "HEAD"))
            self.assertEqual(config, (root / "release-please-config.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
