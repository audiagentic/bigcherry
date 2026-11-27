"""The platform release PR is reconciled with main after an engine release: the pure manifest rule, the head-SHA
guard, a real-git merge, and the workflow wiring."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bigcherry.release import manifest_reconcile as mr  # noqa: E402

_REPO = Path(__file__).resolve().parents[3]


def _manifest(platform: str, llama: str, radiance: str) -> str:
    return ('{\n  "engines/llamacpp": "%s",\n  ".": "%s",\n  "engines/radiance": "%s"\n}\n' % (llama, platform, radiance))


class MergeRuleTests(unittest.TestCase):
    def test_platform_from_head_engines_from_main(self):
        out = mr.merge_manifest_text(_manifest("1.63.0", "11.0.0", "10.0.0"), _manifest("1.62.1", "11.0.0", "10.1.0"), ".")
        self.assertEqual(out, _manifest("1.63.0", "11.0.0", "10.1.0"))

    def test_engine_after_platform_release(self):
        for package, head, expected in (
            ("engines/radiance", _manifest("1.81.1", "11.0.0", "2020.24.0"), _manifest("1.82.0", "11.0.0", "2020.24.0")),
            ("engines/llamacpp", _manifest("1.81.1", "11.1.0", "2020.23.2"), _manifest("1.82.0", "11.1.0", "2020.23.2")),
        ):
            with self.subTest(package=package):
                self.assertEqual(mr.merge_manifest_text(head, _manifest("1.82.0", "11.0.0", "2020.23.2"), package), expected)

    def test_equal_versions_are_allowed(self):
        text = _manifest("1.62.1", "11.0.0", "10.0.0")
        self.assertEqual(mr.merge_manifest_text(text, text, "."), text)

    def test_refuses_missing_package_and_key_mismatch(self):
        good = _manifest("1.63.0", "11.0.0", "10.0.0")
        with self.assertRaises(mr.ManifestReconcileError):
            mr.merge_manifest_text(good, good, "engines/none")
        with self.assertRaises(mr.ManifestReconcileError):
            mr.merge_manifest_text('{".": "1.63.0"}', good, ".")
        with self.assertRaises(mr.ManifestReconcileError):
            mr.merge_manifest_text(good, '{"engines/llamacpp": "1", "engines/radiance": "1"}', ".")

    def test_refuses_downgrades(self):
        # platform on main newer than the PR head
        with self.assertRaises(mr.ManifestReconcileError):
            mr.merge_manifest_text(_manifest("1.62.0", "11.0.0", "10.0.0"), _manifest("1.62.1", "11.0.0", "10.0.0"), ".")
        # an engine on the PR head newer than main would be lowered by taking main's
        with self.assertRaises(mr.ManifestReconcileError):
            mr.merge_manifest_text(_manifest("1.63.0", "11.0.0", "10.2.0"), _manifest("1.62.1", "11.0.0", "10.1.0"), ".")

    def test_refuses_malformed(self):
        good = _manifest("1.63.0", "11.0.0", "10.0.0")
        for bad in ("not json", "[]", _manifest("1.63.0", "11.0.0", "10.0.0-rc1")):
            with self.assertRaises(mr.ManifestReconcileError):
                mr.merge_manifest_text(bad, good, ".")


class FakeGit:
    """Scripted git: records calls, answers by subcommand."""

    def __init__(self, head_sha: str, ancestor: bool = False, push_ok: bool = True):
        self.calls: list[tuple[str, ...]] = []
        self.head_sha, self.ancestor, self.push_ok = head_sha, ancestor, push_ok
        self.head_manifest = _manifest("1.63.0", "11.0.0", "10.0.0")
        self.main_manifest = _manifest("1.62.1", "11.0.0", "10.1.0")

    def __call__(self, cwd, args):
        args = tuple(args)
        self.calls.append(args)
        if args[:2] == ("worktree", "add"):
            Path(args[-2]).mkdir()
        if args[0] == "rev-parse":
            return 0, self.head_sha + "\n", ""
        if args[0] == "merge-base":
            return (0 if self.ancestor else 1) if args[3] == "refs/reconcile/head" else 0, "", ""
        if args[0] == "show":
            return 0, (self.head_manifest if "head" in args[1] else self.main_manifest), ""
        if args[0] == "push":
            return (0 if self.push_ok else 1), "", ""
        return 0, "", ""

    def pushed(self):
        return [c for c in self.calls if c[0] == "push"]


class GuardTests(unittest.TestCase):
    def run_it(self, git, expected="aaa"):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            return mr.reconcile_head(repo, "origin", "main", "rp-branch", expected, ".", git)

    def test_pushes_as_compare_and_swap_on_the_expected_head(self):
        git = FakeGit("aaa")
        self.assertEqual(self.run_it(git), "pushed")
        self.assertEqual(git.pushed()[0], ("push", "-q", "--force-with-lease=refs/heads/rp-branch:aaa", "origin",
                                           "HEAD:refs/heads/rp-branch"))
        self.assertFalse(any(a in ("--force", "-f") or a.startswith("+") for c in git.pushed() for a in c))

    def test_merge_is_built_uncommitted_so_the_rule_always_runs(self):
        git = FakeGit("aaa")
        self.run_it(git)
        merge = next(c for c in git.calls if c[0] == "merge")
        self.assertIn("--no-commit", merge)
        self.assertIn("--no-ff", merge)

    def test_head_moved_before_start_is_a_clean_exit(self):
        git = FakeGit("bbb")
        self.assertEqual(self.run_it(git), "head-moved")
        self.assertEqual(git.pushed(), [])
        self.assertFalse(any(c[0] == "worktree" for c in git.calls))

    def test_rejected_push_is_head_moved(self):
        self.assertEqual(self.run_it(FakeGit("aaa", push_ok=False)), "head-moved")

    def test_rerun_after_reconcile_does_nothing(self):
        git = FakeGit("aaa", ancestor=True)
        self.assertEqual(self.run_it(git), "up-to-date")
        self.assertEqual(git.pushed(), [])
        self.assertFalse(any(c[0] == "merge" for c in git.calls))

    def test_rule_refusal_aborts_before_any_push(self):
        git = FakeGit("aaa")
        git.head_manifest = _manifest("1.63.0", "11.0.0", "10.2.0")  # head engine newer than main: downgrade
        with self.assertRaises(mr.ManifestReconcileError):
            self.run_it(git)
        self.assertEqual(git.pushed(), [])
        self.assertTrue(any(c[:2] == ("merge", "--abort") for c in git.calls))


def _git(cwd, *args):
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args],
                          cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


class RealGitTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        base = Path(self._tmp.name)
        self.origin = base / "origin.git"
        subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(self.origin)], check=True)
        self.repo = base / "work" / "repo"
        self.repo.mkdir(parents=True)
        _git(self.repo, "init", "-q", "-b", "main")
        _git(self.repo, "remote", "add", "origin", str(self.origin))
        _git(self.repo, "config", "user.name", "t")
        _git(self.repo, "config", "user.email", "t@example.invalid")
        (self.repo / mr.MANIFEST).write_text(_manifest("1.62.1", "11.0.0", "10.0.0"))
        (self.repo / "other.txt").write_text("a\n")
        _git(self.repo, "add", "-A")
        _git(self.repo, "commit", "-q", "-m", "base")
        _git(self.repo, "push", "-q", "origin", "main")
        _git(self.repo, "checkout", "-q", "-b", "rp")
        (self.repo / mr.MANIFEST).write_text(_manifest("1.63.0", "11.0.0", "10.0.0"))
        _git(self.repo, "commit", "-qam", "release platform")
        _git(self.repo, "push", "-q", "origin", "rp")
        self.rp_sha = _git(self.repo, "rev-parse", "HEAD")
        _git(self.repo, "checkout", "-q", "main")

    def tearDown(self):
        self._tmp.cleanup()

    def _advance_main(self, manifest: str, other: str | None = None):
        (self.repo / mr.MANIFEST).write_text(manifest)
        if other:
            (self.repo / "other.txt").write_text(other)
        _git(self.repo, "commit", "-qam", "engine release")
        _git(self.repo, "push", "-q", "origin", "main")

    def test_manifest_only_conflict_is_resolved_and_pushed(self):
        self._advance_main(_manifest("1.62.1", "11.0.0", "10.1.0"))
        result = mr.reconcile_head(self.repo, str(self.origin), "main", "rp", self.rp_sha, ".")
        self.assertEqual(result, "pushed")
        merged = _git(self.origin, "show", f"rp:{mr.MANIFEST}")
        self.assertEqual(json.loads(merged), {"engines/llamacpp": "11.0.0", ".": "1.63.0", "engines/radiance": "10.1.0"})
        self.assertEqual(len(_git(self.origin, "rev-list", "--parents", "-n1", "rp").split()), 3)  # a merge commit
        # fresh head, second run: nothing to do
        new = _git(self.origin, "rev-parse", "rp")
        self.assertEqual(mr.reconcile_head(self.repo, str(self.origin), "main", "rp", new, "."), "up-to-date")
        self.assertEqual(_git(self.origin, "rev-parse", "rp"), new)

    def test_other_file_conflict_refuses_and_leaves_branch(self):
        _git(self.repo, "checkout", "-q", "rp")
        (self.repo / "other.txt").write_text("pr\n")
        _git(self.repo, "commit", "-qam", "pr edit")
        _git(self.repo, "push", "-q", "origin", "rp")
        sha = _git(self.repo, "rev-parse", "HEAD")
        _git(self.repo, "checkout", "-q", "main")
        self._advance_main(_manifest("1.62.1", "11.0.0", "10.1.0"), "main\n")
        with self.assertRaises(mr.ManifestReconcileError):
            mr.reconcile_head(self.repo, str(self.origin), "main", "rp", sha, ".")
        self.assertEqual(_git(self.origin, "rev-parse", "rp"), sha)

    def test_clean_merge_with_contradictory_versions_is_refused(self):
        # PR head carries a newer engine than main and main changes an unrelated file: git merges cleanly, the rule refuses
        _git(self.repo, "checkout", "-q", "rp")
        (self.repo / mr.MANIFEST).write_text(_manifest("1.63.0", "11.0.0", "10.2.0"))
        _git(self.repo, "commit", "-qam", "engine ahead")
        _git(self.repo, "push", "-q", "origin", "rp")
        sha = _git(self.repo, "rev-parse", "HEAD")
        _git(self.repo, "checkout", "-q", "main")
        (self.repo / "other.txt").write_text("main\n")
        _git(self.repo, "commit", "-qam", "unrelated")
        _git(self.repo, "push", "-q", "origin", "main")
        with self.assertRaises(mr.ManifestReconcileError):
            mr.reconcile_head(self.repo, str(self.origin), "main", "rp", sha, ".")
        self.assertEqual(_git(self.origin, "rev-parse", "rp"), sha)

    def test_clean_merge_applies_the_rule_result(self):
        self._advance_main(_manifest("1.62.1", "11.0.0", "10.1.0"), "main\n")
        # make the manifest merge clean: the PR head has not touched the manifest lines main changed
        _git(self.repo, "checkout", "-q", "rp")
        (self.repo / "other.txt").write_text("a\n")
        sha = _git(self.repo, "rev-parse", "HEAD")
        _git(self.repo, "checkout", "-q", "main")
        self.assertEqual(mr.reconcile_head(self.repo, str(self.origin), "main", "rp", sha, "."), "pushed")
        merged = json.loads(_git(self.origin, "show", f"rp:{mr.MANIFEST}"))
        self.assertEqual(merged["engines/radiance"], "10.1.0")
        self.assertEqual(merged["."], "1.63.0")

    def test_clean_merge_with_key_set_mismatch_is_refused(self):
        _git(self.repo, "checkout", "-q", "rp")
        (self.repo / mr.MANIFEST).write_text('{\n  "engines/llamacpp": "11.0.0",\n  ".": "1.63.0"\n}\n')
        _git(self.repo, "commit", "-qam", "drop a key")
        _git(self.repo, "push", "-q", "origin", "rp")
        sha = _git(self.repo, "rev-parse", "HEAD")
        _git(self.repo, "checkout", "-q", "main")
        (self.repo / "other.txt").write_text("main\n")
        _git(self.repo, "commit", "-qam", "unrelated")
        _git(self.repo, "push", "-q", "origin", "main")
        with self.assertRaises(mr.ManifestReconcileError):
            mr.reconcile_head(self.repo, str(self.origin), "main", "rp", sha, ".")

    def test_head_moving_to_an_ancestor_after_the_last_read_is_not_overwritten(self):
        self._advance_main(_manifest("1.62.1", "11.0.0", "10.1.0"))
        main_sha = _git(self.repo, "rev-parse", "main")

        def racing(cwd, args):
            if args[0] == "push":  # the branch moves to an ancestor of the candidate after the final read
                _git(self.origin, "update-ref", "refs/heads/rp", main_sha)
            return mr.system_git(cwd, args)

        result = mr.reconcile_head(self.repo, str(self.origin), "main", "rp", self.rp_sha, ".", racing)
        self.assertEqual(result, "head-moved")
        self.assertEqual(_git(self.origin, "rev-parse", "rp"), main_sha)

    def test_other_file_conflict_is_named_before_any_commit(self):
        _git(self.repo, "checkout", "-q", "rp")
        (self.repo / "other.txt").write_text("pr\n")
        _git(self.repo, "commit", "-qam", "pr edit")
        _git(self.repo, "push", "-q", "origin", "rp")
        sha = _git(self.repo, "rev-parse", "HEAD")
        _git(self.repo, "checkout", "-q", "main")
        self._advance_main(_manifest("1.62.1", "11.0.0", "10.1.0"), "main\n")
        with self.assertRaisesRegex(mr.ManifestReconcileError, "other.txt"):
            mr.reconcile_head(self.repo, str(self.origin), "main", "rp", sha, ".")

    def test_stale_expected_sha_pushes_nothing(self):
        self._advance_main(_manifest("1.62.1", "11.0.0", "10.1.0"))
        self.assertEqual(mr.reconcile_head(self.repo, str(self.origin), "main", "rp", "0" * 40, "."), "head-moved")
        self.assertEqual(_git(self.origin, "rev-parse", "rp"), self.rp_sha)


_CONFIG = {"packages": {".": {"component": "bc-platform"}, "engines/radiance": {"component": "bc-radiance"}}}
_BRANCH = "release-please--branches--main--components--bc-platform"


def _pr(**over):
    pr = {"number": 7, "state": "OPEN", "baseRefName": "main", "labels": [{"name": mr.LABEL}], "headRefName": _BRANCH,
          "isCrossRepository": False, "isDraft": False, "headRefOid": "aaa", "mergeable": "CONFLICTING", "mergeStateStatus": "DIRTY"}
    pr.update(over)
    return pr


class SelectionTests(unittest.TestCase):
    def run_it(self, listed, views, results=("pushed",)):
        views, results = list(views), list(results)
        writes, sleeps = [], []

        def gh(*args):
            return listed if args[1] == "list" else views.pop(0) if len(views) > 1 else views[0]

        def git(cwd, args):  # only reconcile_head's own git use is replaced by the patch below
            raise AssertionError(args)

        def fake_reconcile(repo, remote, base, head, sha, package, git_):
            writes.append((head, sha, package))
            return results.pop(0)

        real = mr.reconcile_head
        mr.reconcile_head = fake_reconcile
        try:
            report = mr.reconcile_platform(Path("."), "url", _CONFIG, gh, git, sleeps.append, attempts=3, seconds=1)
        finally:
            mr.reconcile_head = real
        return report, writes, sleeps

    def test_exact_same_repo_platform_pr_is_written_with_its_read_sha(self):
        report, writes, _ = self.run_it([_pr()], [_pr(headRefOid="bbb")])
        self.assertEqual(report, ["PR #7: pushed"])
        self.assertEqual(writes, [(_BRANCH, "bbb", ".")])

    def test_unknown_then_conflicting_retries_then_writes(self):
        report, writes, sleeps = self.run_it([_pr()], [_pr(mergeable="UNKNOWN", mergeStateStatus="UNKNOWN"), _pr()])
        self.assertEqual(report, ["PR #7: pushed"])
        self.assertEqual(sleeps, [1])

    def test_unknown_then_mergeable_does_nothing(self):
        clean = _pr(mergeable="MERGEABLE", mergeStateStatus="CLEAN")
        report, writes, _ = self.run_it([_pr()], [_pr(mergeable="UNKNOWN"), clean])
        self.assertEqual(report, ["PR #7: no conflict, nothing to do"])
        self.assertEqual(writes, [])

    def test_perpetual_unknown_is_a_named_deferred_state(self):
        report, writes, sleeps = self.run_it([_pr()], [_pr(mergeable="UNKNOWN", mergeStateStatus="UNKNOWN")])
        self.assertEqual(report, ["PR #7: deferred-mergeability-unknown"])
        self.assertEqual((writes, len(sleeps)), ([], 3))

    def test_head_moving_during_reconcile_is_retried_then_deferred(self):
        report, writes, _ = self.run_it([_pr()], [_pr(), _pr(headRefOid="bbb"), _pr(headRefOid="ccc")],
                                        results=("head-moved", "head-moved", "head-moved"))
        self.assertEqual(report, ["PR #7: deferred-head-moved"])
        self.assertEqual([w[1] for w in writes], ["aaa", "bbb", "ccc"])

    def test_head_moved_then_success(self):
        report, writes, _ = self.run_it([_pr()], [_pr(), _pr(headRefOid="bbb")], results=("head-moved", "pushed"))
        self.assertEqual(report, ["PR #7: pushed"])

    def test_other_prs_are_left_alone(self):
        cases = [
            _pr(headRefName="x-" + _BRANCH), _pr(headRefName=_BRANCH + "-2"),   # another branch
            _pr(isCrossRepository=True), _pr(isDraft=True), _pr(baseRefName="release"), _pr(state="CLOSED"), _pr(labels=[]),
        ]
        for case in cases:
            with self.subTest(case=case):
                report, writes, _ = self.run_it([case], [case])
                self.assertEqual((report, writes), ([], []))

    def test_pr_that_closed_between_list_and_view_is_refused(self):
        report, writes, _ = self.run_it([_pr()], [_pr(state="CLOSED")])
        self.assertEqual(writes, [])
        self.assertTrue(report[0].startswith("PR #7: refused"))


class AllComponentsTests(unittest.TestCase):
    def test_config_drives_all_branches(self):
        from unittest.mock import patch
        config = {"packages": {".": {"component": "bc-platform"},
                               "engines/radiance": {"component": "bc-radiance"},
                               "engines/llamacpp": {"component": "bc-llamacpp"}}}
        with patch.object(mr, "reconcile_component", return_value=["done"]) as run:
            self.assertEqual(mr.reconcile_all(Path("."), "origin", config), ["done"] * 3)
            self.assertEqual([c.args[2:] for c in run.call_args_list],
                             [(".", "bc-platform"), ("engines/radiance", "bc-radiance"),
                              ("engines/llamacpp", "bc-llamacpp")])

    def test_duplicate_components_refused_before_write(self):
        from unittest.mock import patch
        config = {"packages": {".": {"component": "same"}, "other": {"component": "same"}}}
        with patch.object(mr, "reconcile_component") as run:
            with self.assertRaises(mr.ManifestReconcileError):
                mr.reconcile_all(Path("."), "origin", config)
            run.assert_not_called()


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.wf = yaml.safe_load((_REPO / ".github/workflows/release-please.yml").read_text(encoding="utf-8"))
        steps = self.wf["jobs"]["release-please"]["steps"]
        self.step = next(s for s in steps if "manifest_reconcile" in s.get("run", ""))
        self.steps = steps

    def test_step_runs_after_release_please_with_the_existing_token(self):
        self.assertGreater(self.steps.index(self.step), 0)
        self.assertEqual(self.step["env"]["GH_TOKEN"], "${{ secrets.RELEASE_PLEASE_TOKEN }}")

    def test_permissions_and_trigger_cannot_loop(self):
        on = self.wf.get("on", self.wf.get(True))
        self.assertEqual(on, {"push": {"branches": ["main"]}})  # only main; the step pushes a release branch
        self.assertEqual(self.wf["permissions"], {"contents": "write", "pull-requests": "write"})
        self.assertNotIn("gh pr merge", self.step["run"])

    def test_runs_are_serialised(self):
        self.assertFalse(self.wf["concurrency"]["cancel-in-progress"])


if __name__ == "__main__":
    unittest.main()
