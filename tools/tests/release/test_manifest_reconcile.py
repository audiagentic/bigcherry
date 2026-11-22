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

    def __init__(self, head_sha: str, ancestor: bool = False, push_ok: bool = True, later_sha: str | None = None):
        self.calls: list[tuple[str, ...]] = []
        self.head_sha, self.ancestor, self.push_ok = head_sha, ancestor, push_ok
        self.later_sha = later_sha
        self.rev_parses = 0

    def __call__(self, cwd, args):
        args = tuple(args)
        self.calls.append(args)
        if args[0] == "rev-parse":
            self.rev_parses += 1
            sha = self.head_sha if self.rev_parses == 1 or self.later_sha is None else self.later_sha
            return 0, sha + "\n", ""
        if args[0] == "merge-base":
            return (0 if self.ancestor else 1), "", ""
        if args[0] == "merge":
            return 0, "", ""
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

    def test_pushes_plain_when_head_unchanged(self):
        git = FakeGit("aaa")
        self.assertEqual(self.run_it(git), "pushed")
        push = git.pushed()[0]
        self.assertEqual(push, ("push", "-q", "origin", "HEAD:refs/heads/rp-branch"))
        self.assertFalse(any("--force" in a or a.startswith("+") for c in git.calls if c[0] == "push" for a in c))

    def test_head_moved_before_start_is_a_clean_exit(self):
        git = FakeGit("bbb")
        self.assertEqual(self.run_it(git), "head-moved")
        self.assertEqual(git.pushed(), [])

    def test_head_moved_during_merge_is_not_pushed(self):
        git = FakeGit("aaa", later_sha="ccc")
        self.assertEqual(self.run_it(git), "head-moved")
        self.assertEqual(git.pushed(), [])

    def test_rejected_push_is_head_moved(self):
        self.assertEqual(self.run_it(FakeGit("aaa", push_ok=False)), "head-moved")

    def test_rerun_after_reconcile_does_nothing(self):
        git = FakeGit("aaa", ancestor=True)
        self.assertEqual(self.run_it(git), "up-to-date")
        self.assertEqual(git.pushed(), [])
        self.assertFalse(any(c[0] == "merge" for c in git.calls))

    def test_never_updates_main(self):
        with self.assertRaises(mr.ManifestReconcileError):
            mr.reconcile_head(Path("."), "origin", "main", "main", "aaa", ".", FakeGit("aaa"))


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

    def test_stale_expected_sha_pushes_nothing(self):
        self._advance_main(_manifest("1.62.1", "11.0.0", "10.1.0"))
        self.assertEqual(mr.reconcile_head(self.repo, str(self.origin), "main", "rp", "0" * 40, "."), "head-moved")
        self.assertEqual(_git(self.origin, "rev-parse", "rp"), self.rp_sha)


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.wf = yaml.safe_load((_REPO / ".github/workflows/release-please.yml").read_text(encoding="utf-8"))
        steps = self.wf["jobs"]["release-please"]["steps"]
        self.step = next(s for s in steps if "manifest_reconcile" in s.get("run", ""))
        self.steps = steps

    def test_step_runs_after_release_please_with_the_existing_token(self):
        self.assertGreater(self.steps.index(self.step), 0)
        self.assertEqual(self.step["env"]["GH_TOKEN"], "${{ secrets.RELEASE_PLEASE_TOKEN }}")
        self.assertNotIn("secrets.", json.dumps(self.step).replace("secrets.RELEASE_PLEASE_TOKEN", ""))

    def test_permissions_and_trigger_cannot_loop(self):
        on = self.wf.get("on", self.wf.get(True))
        self.assertEqual(on, {"push": {"branches": ["main"]}})  # only main; the step pushes a release branch
        self.assertEqual(self.wf["permissions"], {"contents": "write", "pull-requests": "write"})
        self.assertNotIn("gh pr merge", self.step["run"])
        self.assertNotIn("--force", self.step["run"])

    def test_runs_are_serialised(self):
        self.assertFalse(self.wf["concurrency"]["cancel-in-progress"])


if __name__ == "__main__":
    unittest.main()
