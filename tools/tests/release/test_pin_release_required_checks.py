"""pin-release required-check and PR-update behavior."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.release import pin_release as _pr


def _result(stdout: str = "", stderr: str = "", returncode: int = 0) -> SimpleNamespace:
    return SimpleNamespace(stdout=stdout, stderr=stderr, returncode=returncode)


class RequiredChecksTests(unittest.TestCase):
    def setUp(self):
        self.plan = SimpleNamespace(repo_root=Path("/repo"), main="main", remote="origin")
        self.pr = {
            "number": 17,
            "url": "https://example.invalid/pr/17",
            "headRefName": "release-please--branches--main--components--bc-llamacpp",
        }

    def test_branch_up_to_date_needs_no_update(self):
        state = {"mergeStateStatus": "CLEAN", "mergeable": "MERGEABLE", "headRefName": self.pr["headRefName"]}
        with mock.patch.object(_pr, "_pr_state", return_value=state), mock.patch.object(_pr, "_run") as run:
            self.assertFalse(_pr._bring_pr_up_to_main(self.plan, self.pr, "release"))
        run.assert_not_called()

    def test_behind_branch_uses_update_branch(self):
        state = {"mergeStateStatus": "BEHIND", "mergeable": "MERGEABLE", "headRefName": self.pr["headRefName"]}
        with mock.patch.object(_pr, "_pr_state", return_value=state), \
                mock.patch.object(_pr, "_run", return_value=_result()) as run:
            self.assertTrue(_pr._bring_pr_up_to_main(self.plan, self.pr, "release"))
        run.assert_called_once_with(Path("/repo"), "gh", "pr", "update-branch", "17", check=False)

    def test_manifest_conflict_keeps_release_package_and_main_other_package_in_both_orders(self):
        release_first = (
            '{\n'
            '  "engines/llamacpp": "11552.2.1",\n'
            '  ".": "1.1.0"\n'
            '}\n'
        )
        main_first = (
            '{\n'
            '  "engines/llamacpp": "11552.1.1",\n'
            '  ".": "1.2.0"\n'
            '}\n'
        )
        self.assertEqual(
            _pr._merge_manifest_text(release_first, main_first),
            '{\n  "engines/llamacpp": "11552.2.1",\n  ".": "1.2.0"\n}\n',
        )

        release_other_first = (
            '{\n'
            '  ".": "1.1.0",\n'
            '  "engines/llamacpp": "11552.2.1"\n'
            '}\n'
        )
        main_other_first = (
            '{\n'
            '  ".": "1.2.0",\n'
            '  "engines/llamacpp": "11552.1.1"\n'
            '}\n'
        )
        self.assertEqual(
            _pr._merge_manifest_text(release_other_first, main_other_first),
            '{\n  ".": "1.2.0",\n  "engines/llamacpp": "11552.2.1"\n}\n',
        )

    def test_other_conflict_fails_closed_and_names_files(self):
        with tempfile.TemporaryDirectory() as td:
            plan = SimpleNamespace(repo_root=Path(td) / "repo", main="main", remote="origin")
            plan.repo_root.mkdir()
            state = {"headRefName": self.pr["headRefName"]}

            def fake_git(root, *args, check=True):
                if args[:3] == ("diff", "--name-only", "--diff-filter=U"):
                    return ".release-please-manifest.json\nother.txt"
                return ""

            def fake_run(root, *cmd, check=True, timeout=600):
                if cmd[:2] == ("git", "merge") and "--abort" not in cmd:
                    return _result(returncode=1)
                return _result()

            with mock.patch.object(_pr, "_pr_state", return_value=state), \
                    mock.patch.object(_pr, "_git", side_effect=fake_git), \
                    mock.patch.object(_pr, "_run", side_effect=fake_run):
                with self.assertRaisesRegex(_pr.PinReleaseError, r"other\.txt"):
                    _pr._merge_pr_branch_locally(plan, self.pr, "release")

    def test_checks_pending_then_pass(self):
        pending = json.dumps([
            {"name": "conventional-title", "state": "PENDING", "link": "u/title"},
            {"name": "index-line-endings", "state": "SUCCESS", "link": "u/endings"},
        ])
        passed = json.dumps([
            {"name": "conventional-title", "state": "SUCCESS", "link": "u/title"},
            {"name": "index-line-endings", "state": "SUCCESS", "link": "u/endings"},
        ])
        with mock.patch.object(_pr, "_required_checks", return_value=("conventional-title", "index-line-endings")), \
                mock.patch.object(_pr, "_run", side_effect=[_result(pending, returncode=8), _result(passed)]), \
                mock.patch.object(_pr.time, "sleep"):
            _pr._wait_required_checks(self.plan, self.pr, "release", wait_s=10, poll_s=0, grace_s=10)

    def test_failing_required_check_names_check_and_url(self):
        failed = json.dumps([
            {"name": "conventional-title", "state": "FAILURE", "link": "https://ci.invalid/title"},
        ])
        with mock.patch.object(_pr, "_required_checks", return_value=("conventional-title",)), \
                mock.patch.object(_pr, "_run", return_value=_result(failed)):
            with self.assertRaisesRegex(
                _pr.PinReleaseError,
                r"conventional-title failed: https://ci\.invalid/title",
            ):
                _pr._wait_required_checks(self.plan, self.pr, "release", wait_s=10, poll_s=0)

    def test_no_checks_reported_triggers_once_then_waits(self):
        passed = json.dumps([
            {"name": "conventional-title", "state": "SUCCESS", "link": "u/title"},
            {"name": "index-line-endings", "state": "SUCCESS", "link": "u/endings"},
        ])
        with mock.patch.object(_pr, "_required_checks", return_value=("conventional-title", "index-line-endings")), \
                mock.patch.object(_pr, "_run", side_effect=[_result("[]"), _result(passed)]), \
                mock.patch.object(_pr, "_trigger_pr_checks") as trigger, \
                mock.patch.object(_pr.time, "sleep"):
            _pr._wait_required_checks(self.plan, self.pr, "release", wait_s=10, poll_s=0, grace_s=0)
        trigger.assert_called_once_with(self.plan, self.pr, "release", print)

    def test_up_to_date_trigger_closes_and_reopens_once(self):
        state = {"mergeStateStatus": "CLEAN", "mergeable": "MERGEABLE"}
        lines = []
        with mock.patch.object(_pr, "_pr_state", return_value=state), \
                mock.patch.object(_pr, "_run", side_effect=[_result(), _result()]) as run:
            _pr._trigger_pr_checks(self.plan, self.pr, "release", lines.append)
        self.assertEqual(
            [call.args[1:5] for call in run.call_args_list],
            [("gh", "pr", "close", "17"), ("gh", "pr", "reopen", "17")],
        )
        self.assertTrue(any("closed and reopened" in line for line in lines))

    def test_dry_run_describes_update_and_required_checks_without_writes(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "releases").mkdir()
            (root / "release-please-config.json").write_text(
                json.dumps({"packages": {"engines/llamacpp": {}, ".": {}}}, indent=2) + "\n",
                encoding="utf-8",
            )
            (root / ".release-please-manifest.json").write_text(
                json.dumps({"engines/llamacpp": "11552.1.1", ".": "1.1.0"}, indent=2) + "\n",
                encoding="utf-8",
            )
            plan = SimpleNamespace(
                repo_root=root,
                llama_tag="b11552",
                release_tag="bc-llamacpp-11552.2.1",
                version="11552.2.1",
                branch="fix/release",
                main="main",
                remote="origin",
            )
            lines = []
            pr = {"number": 17, "title": "release", "headRefName": self.pr["headRefName"]}
            with mock.patch.object(_pr, "_tag_exists", return_value=False), \
                    mock.patch.object(_pr, "_git", return_value=""), \
                    mock.patch.object(_pr, "_pr_for_head", return_value=pr), \
                    mock.patch.object(_pr, "_release_pr", return_value=pr), \
                    mock.patch.object(_pr, "_required_checks", return_value=("conventional-title", "index-line-endings")), \
                    mock.patch.object(_pr, "_run") as run:
                _pr._dry_run(plan, _pr.PHASES.index("release"), lines.append)
            run.assert_not_called()
            self.assertTrue(any("will merge main if behind" in line for line in lines))
            self.assertTrue(any("conventional-title, index-line-endings" in line for line in lines))


if __name__ == "__main__":
    unittest.main()
