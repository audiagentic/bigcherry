"""pin-release required-check and PR-update behavior."""

from __future__ import annotations

import itertools
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
            _pr._merge_manifest_text(release_first, main_first, "engines/llamacpp"),
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
            _pr._merge_manifest_text(release_other_first, main_other_first, "engines/llamacpp"),
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
                package="engines/llamacpp",
                engine="llamacpp",
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


class FakeGh:
    """A fake `gh` for the release PR: the bot-created PR reports no checks until it is closed and reopened by the account."""

    def __init__(self, *, checks_after_reopen: str, merge_state: str = "CLEAN"):
        self.calls: list[tuple[str, ...]] = []
        self.reopened = False
        self.checks_after_reopen = checks_after_reopen
        self.merge_state = merge_state

    def __call__(self, root, *cmd, check=True, timeout=600):
        self.calls.append(cmd)
        sub = cmd[1:3]
        if sub == ("pr", "view"):
            return _result(json.dumps({"mergeStateStatus": self.merge_state, "mergeable": "MERGEABLE", "state": "OPEN",
                                       "url": "u/17", "headRefName": "release-please--branches--main--components--bc-radiance"}))
        if sub == ("pr", "checks"):
            if not self.reopened:
                return _result("", "no checks reported on the release branch", 1)
            return _result(self.checks_after_reopen)
        if sub == ("pr", "close"):
            return _result()
        if sub == ("pr", "reopen"):
            self.reopened = True
            return _result()
        if sub == ("pr", "merge"):
            return _result()
        raise AssertionError(f"unexpected gh call {cmd}")

    def verbs(self) -> list[str]:
        return [c[2] for c in self.calls if c[1] == "pr"]


class ReleasePrChecksTests(unittest.TestCase):
    def setUp(self):
        self.plan = SimpleNamespace(repo_root=Path("/repo"), main="main", remote="origin", package="engines/radiance")
        self.pr = {"number": 17, "url": "u/17", "headRefName": "release-please--branches--main--components--bc-radiance"}
        required = mock.patch.object(_pr, "_required_checks", return_value=("conventional-title", "index-line-endings"))
        required.start()
        self.addCleanup(required.stop)
        sleep = mock.patch.object(_pr.time, "sleep")
        sleep.start()
        self.addCleanup(sleep.stop)

    @staticmethod
    def _passed():
        return json.dumps([{"name": "conventional-title", "state": "SUCCESS", "link": "l"},
                           {"name": "index-line-endings", "state": "SUCCESS", "link": "l"}])

    def test_bot_created_release_pr_without_checks_is_reopened_then_merged_after_the_checks_pass(self):
        gh = FakeGh(checks_after_reopen=self._passed())
        lines = []
        with mock.patch.object(_pr, "_run", gh), mock.patch.object(_pr.time, "time", side_effect=itertools.count(0, 30)):
            _pr._wait_and_merge_pr(self.plan, self.pr, "release", wait_s=600, log=lines.append)
        verbs = gh.verbs()
        self.assertEqual([v for v in verbs if v in ("close", "reopen", "merge")], ["close", "reopen", "merge"])
        self.assertLess(verbs.index("reopen"), verbs.index("merge"))
        self.assertTrue(any("closed and reopened" in line for line in lines))

    def test_a_failing_check_after_the_trigger_blocks_the_merge(self):
        failed = json.dumps([{"name": "conventional-title", "state": "FAILURE", "link": "https://ci/x"}])
        gh = FakeGh(checks_after_reopen=failed)
        with mock.patch.object(_pr, "_run", gh), mock.patch.object(_pr.time, "time", side_effect=itertools.count(0, 30)):
            with self.assertRaisesRegex(_pr.PinReleaseError, "conventional-title failed"):
                _pr._wait_and_merge_pr(self.plan, self.pr, "release", wait_s=600, log=lambda *_: None)
        self.assertNotIn("merge", gh.verbs())

    def test_checks_that_never_arrive_stop_without_merging_and_trigger_only_once(self):
        gh = FakeGh(checks_after_reopen="[]")
        with mock.patch.object(_pr, "_run", gh), mock.patch.object(_pr.time, "time", side_effect=itertools.count(0, 7)):
            with self.assertRaisesRegex(_pr.PinReleaseError, "did not pass within"):
                _pr._wait_and_merge_pr(self.plan, self.pr, "release", wait_s=60, log=lambda *_: None)
        self.assertEqual(gh.verbs().count("close"), 1)
        self.assertNotIn("merge", gh.verbs())

    def test_release_pr_files_are_verified_before_anything_is_written_to_it(self):
        gh = FakeGh(checks_after_reopen=self._passed())
        problem = _pr.PinReleaseError("release", "unexpected files")
        with mock.patch.object(_pr, "_run", gh):
            with self.assertRaisesRegex(_pr.PinReleaseError, "unexpected files"):
                _pr._wait_and_merge_pr(self.plan, self.pr, "release", wait_s=60, verify=mock.Mock(side_effect=problem))
        self.assertEqual(gh.calls, [])


def _release_plan():
    return SimpleNamespace(repo_root=Path("/repo"), llama_tag="b1", release_tag="bc-1.0.0", version="1.0.0",
                           branch="work", main="main", engine="radiance", package="engines/radiance")


class ConfirmReleaseTests(unittest.TestCase):
    def setUp(self):
        self.plan = SimpleNamespace(release_tag="bc-radiance-170.0.0", version="170.0.0", main="main")

    def test_yes_confirms_without_asking(self):
        ask = mock.Mock()
        _pr.confirm_release(self.plan, True, lambda *_: None, ask, interactive=False)
        ask.assert_not_called()

    def test_without_a_terminal_and_without_yes_it_stops_before_tagging(self):
        with self.assertRaisesRegex(_pr.PinReleaseError, r"stopped before tagging bc-radiance-170\.0\.0.*--yes") as ctx:
            _pr.confirm_release(self.plan, False, lambda *_: None, mock.Mock(), interactive=False)
        self.assertEqual(ctx.exception.phase, "release")

    def test_a_terminal_is_asked_once_and_only_y_confirms(self):
        for answer, confirmed in (("y", True), ("YES", True), ("", False), ("n", False)):
            ask = mock.Mock(return_value=answer)
            if confirmed:
                _pr.confirm_release(self.plan, False, lambda *_: None, ask, interactive=True)
            else:
                with self.assertRaisesRegex(_pr.PinReleaseError, "not confirmed"):
                    _pr.confirm_release(self.plan, False, lambda *_: None, ask, interactive=True)
            ask.assert_called_once()


class RunConfirmationTests(unittest.TestCase):
    def _run_through(self, *, yes, interactive, tagged=False, through="sync"):
        order, lines = [], []
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            (repo / "releases").mkdir()
            (repo / "releases" / "b1.json").write_text(json.dumps({"notes": "release 1.0.0"}), encoding="utf-8")

            def phase(name):
                return lambda *a, **k: order.append(name) or True

            with mock.patch.object(_pr, "make_plan", return_value=_release_plan()), \
                    mock.patch.object(_pr, "_tag_exists", return_value=tagged), \
                    mock.patch.object(_pr, "phase_record", side_effect=phase("record")), \
                    mock.patch.object(_pr, "phase_notes", side_effect=phase("notes")), \
                    mock.patch.object(_pr, "phase_main", side_effect=phase("main")), \
                    mock.patch.object(_pr, "phase_release", side_effect=phase("release")), \
                    mock.patch.object(_pr, "phase_unpin", side_effect=phase("unpin")), \
                    mock.patch.object(_pr, "phase_sync", side_effect=phase("sync")):
                try:
                    _pr.run(repo, "radiance", "b1", "", through=through, log=lines.append, yes=yes,
                            ask=lambda *_: "n", interactive=interactive)
                    error = None
                except _pr.PinReleaseError as exc:
                    error = exc
        return order, lines, error

    def test_without_terminal_or_yes_the_stages_before_the_tag_finish_and_it_stops(self):
        order, lines, error = self._run_through(yes=False, interactive=False)
        self.assertEqual(order, ["record", "notes", "main"])
        self.assertEqual(error.phase, "release")
        self.assertTrue(any("main" in line and "done" in line for line in lines))

    def test_with_yes_it_goes_all_the_way(self):
        order, _, error = self._run_through(yes=True, interactive=False)
        self.assertIsNone(error)
        self.assertEqual(order, ["record", "notes", "main", "release", "unpin", "sync"])

    def test_an_already_tagged_release_asks_nothing(self):
        order, _, error = self._run_through(yes=False, interactive=False, tagged=True)
        self.assertIsNone(error)
        self.assertEqual(order[-3:], ["release", "unpin", "sync"])

    def test_stopping_through_main_asks_nothing(self):
        order, _, error = self._run_through(yes=False, interactive=False, through="main")
        self.assertIsNone(error)
        self.assertEqual(order, ["record", "notes", "main"])


if __name__ == "__main__":
    unittest.main()
