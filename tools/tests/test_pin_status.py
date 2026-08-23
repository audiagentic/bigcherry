"""RE48 pin-consistency guard: the verdict matrix and policy tiers.

The fake-tree fixtures reproduce the incident shapes exactly:
- S1: vendor at a release-audited revision, pin elsewhere, NO marker ->
  drift (a record is evidence, not a transition).
- committed marker, pin==to, vendor==from, base audited -> mid-rebase.
- marker written but not committed -> uncommitted-transition.
- repin writes the marker atomically with the pin move (uncommitted).
- pull refuses while the marker is uncommitted.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bigcherry import pin_status  # noqa: E402
from bigcherry import pin_transition  # noqa: E402

_PIN_LINE = re.compile(r'^pinned\s*=\s*"([^"]+)"', re.MULTILINE)

# Root "init" commits in different fake trees (_fake_tree's own commit, plus
# _upstream's b1/b2) must be byte-for-byte reproducible commit objects when
# their tree content matches, or tests that assert two independently-created
# trees resolve to the SAME commit hash (e.g. RemoteAndAggregateTests
# comparing a fake "local" tree's HEAD against a separate fake "campaign"
# tree's HEAD) become timestamp-dependent: git commit hashes include
# author/committer timestamps at 1-second resolution, so two `git commit`
# calls landing in different wall-clock seconds -- easy under a loaded full
# suite run -- silently produce different hashes for identical content. Real
# flaky failure hit this session; root-caused with GPT (req_37641bb91ff4442c).
_GIT_COMMIT_ENV = {
    "GIT_AUTHOR_NAME": "pin-status-test",
    "GIT_AUTHOR_EMAIL": "pin-status-test@example.com",
    "GIT_AUTHOR_DATE": "2000-01-01T00:00:00+0000",
    "GIT_COMMITTER_NAME": "pin-status-test",
    "GIT_COMMITTER_EMAIL": "pin-status-test@example.com",
    "GIT_COMMITTER_DATE": "2000-01-01T00:00:00+0000",
}


def _git(root: Path, *args: str) -> str:
    env = os.environ.copy()
    env.update(_GIT_COMMIT_ENV)
    result = subprocess.run(
        ["git", "-C", str(root), "-c", "user.email=pin-status-test@example.com",
         "-c", "user.name=pin-status-test",
         "-c", "commit.gpgSign=false",
         *args],
        check=True, capture_output=True, text=True, env=env,
    )
    return result.stdout.strip()


def _upstream(tmp: Path) -> Path:
    up = tmp / "upstream"
    up.mkdir()
    _git(up, "init", "-q")
    (up / "file.txt").write_text("v1", encoding="utf-8")
    _git(up, "add", ".")
    _git(up, "commit", "-q", "-m", "b1")
    _git(up, "tag", "b1")
    (up / "file.txt").write_text("v2", encoding="utf-8")
    _git(up, "add", ".")
    _git(up, "commit", "-q", "-m", "b2")
    _git(up, "tag", "b2")
    return up


def _fake_tree(tmp: Path, name: str, upstream: Path, pin: str, at_tag: str,
               records: dict[str, str] | None = None) -> Path:
    """A minimal bigcherry-shaped tree: repo + vendor/llama.cpp (a clone of
    `upstream` at `at_tag`) + releases/ records."""
    root = tmp / name
    (root / "config").mkdir(parents=True)
    (root / "vendor").mkdir(parents=True)
    _git(root, "init", "-q")
    (root / "config" / "recipes.toml").write_text(
        f'pinned = "{pin}"\n', encoding="utf-8")
    _git(root, "add", "config")
    _git(root, "commit", "-q", "-m", "init")
    subprocess.run(
        ["git", "-C", str(root / "vendor"), "clone", "-q",
         str(upstream), "llama.cpp"],
        check=True, capture_output=True)
    vendor = root / "vendor" / "llama.cpp"
    _git(vendor, "checkout", "-q", at_tag)
    releases = root / "releases"
    releases.mkdir()
    (releases / "index.json").write_text("[]", encoding="utf-8")
    for slug, stage in (records or {}).items():
        tag_sha = _git(vendor, "rev-parse", f"{slug}^{{commit}}")
        (releases / f"{slug}.json").write_text(
            json.dumps({"slug": slug, "revision": tag_sha, "stage": stage}),
            encoding="utf-8")
    return root


def _paths_for(root: Path) -> pin_status.RepoPaths:
    return pin_status.RepoPaths(
        repo_root=root,
        llama_root=root / "vendor" / "llama.cpp",
        releases_dir=root / "releases",
        artifacts_dir=root / "artifacts",
    )


def _write_marker(root: Path, from_tag: str, to_tag: str,
                  commit: bool) -> None:
    vendor = root / "vendor" / "llama.cpp"
    marker = pin_transition.write(
        _git(vendor, "rev-parse", f"{from_tag}^{{commit}}"),
        _git(vendor, "rev-parse", f"{to_tag}^{{commit}}"),
        to_tag,
        _git(root, "rev-parse", "HEAD"),
        root / "releases" / "pin-transition.json",
    )
    assert marker is not None
    if commit:
        _git(root, "add", "releases/pin-transition.json")
        _git(root, "commit", "-q", "-m", "rebase: declare transition")


def _status(root: Path) -> pin_status.LocalStatus:
    return pin_status.local_status(_paths_for(root))


class VerdictMatrixTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.upstream = _upstream(self.tmp)

    def tearDown(self):
        self._tmp.cleanup()

    def test_consistent(self):
        root = _fake_tree(self.tmp, "t", self.upstream, "b2", "b2")
        status = _status(root)
        self.assertEqual(status.verdict, "consistent")
        self.assertIsNone(status.marker)

    def test_s1_shape_is_drift_not_mid_rebase(self):
        # vendor at audited b1, pin moved to b2, NO marker: the S1 incident.
        root = _fake_tree(self.tmp, "t", self.upstream, "b2", "b1",
                          records={"b1": "audited"})
        status = _status(root)
        self.assertEqual(status.verdict, "drift")
        self.assertTrue(any("records are evidence" in r
                            for r in status.reasons))

    def test_uncommitted_transition(self):
        root = _fake_tree(self.tmp, "t", self.upstream, "b2", "b1",
                          records={"b1": "audited"})
        _write_marker(root, "b1", "b2", commit=False)
        status = _status(root)
        self.assertEqual(status.verdict, "uncommitted-transition")
        self.assertEqual(status.marker_state, "uncommitted")

    def test_committed_mid_rebase(self):
        root = _fake_tree(self.tmp, "t", self.upstream, "b2", "b1",
                          records={"b1": "audited"})
        _write_marker(root, "b1", "b2", commit=True)
        status = _status(root)
        self.assertEqual(status.verdict, "mid-rebase")
        self.assertEqual(status.marker_state, "committed-clean")

    def test_stale_marker_is_drift(self):
        # marker declares -> b2, but the pin has since moved to b3.
        _git(self.upstream, "commit", "--allow-empty", "-q", "-m", "b3")
        _git(self.upstream, "tag", "b3")
        root = _fake_tree(self.tmp, "t", self.upstream, "b3", "b1",
                          records={"b1": "audited"})
        _git(root / "vendor" / "llama.cpp", "fetch", "-q", "origin")
        _write_marker(root, "b1", "b2", commit=True)
        status = _status(root)
        self.assertEqual(status.verdict, "drift")
        self.assertTrue(any("stale marker" in r for r in status.reasons))

    def test_broken_base_is_drift(self):
        root = _fake_tree(self.tmp, "t", self.upstream, "b2", "b1",
                          records={"b1": "broken"})
        _write_marker(root, "b1", "b2", commit=True)
        status = _status(root)
        self.assertEqual(status.verdict, "drift")
        self.assertTrue(any("broken" in r for r in status.reasons))

    def test_unavailable(self):
        root = _fake_tree(self.tmp, "t", self.upstream, "b2", "b2")
        status = pin_status.local_status(pin_status.RepoPaths(
            repo_root=root,
            llama_root=root / "nope",
            releases_dir=root / "releases",
            artifacts_dir=root / "artifacts",
        ))
        self.assertEqual(status.verdict, "unavailable")

    def test_unresolvable_pin(self):
        root = _fake_tree(self.tmp, "t", self.upstream, "b9", "b2")
        status = _status(root)
        self.assertEqual(status.verdict, "unresolvable-pin")


class RemoteAndAggregateTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.upstream = _upstream(self.tmp)
        self.local_root = _fake_tree(self.tmp, "a", self.upstream, "b2", "b2")

    def tearDown(self):
        self._tmp.cleanup()

    def _probe(self):
        def probe(alias: str, path: str):
            root = Path(path)
            pin = _PIN_LINE.search(
                (root / "config" / "recipes.toml").read_text(encoding="utf-8"))
            return (_git(root / "vendor" / "llama.cpp", "rev-parse", "HEAD"),
                    pin.group(1) if pin else None,
                    _git(root, "rev-parse", "HEAD"))
        return probe

    def _remote_status(self, name: str, pin: str, at_tag: str,
                       probe=None) -> pin_status.RemoteStatus:
        root = _fake_tree(self.tmp, name, self.upstream, pin, at_tag)
        return pin_status.remote_status(
            name, "alias", str(root),
            pin_status.local_status(_paths_for(self.local_root)),
            self.local_root / "vendor" / "llama.cpp",
            probe=probe or self._probe(),
        )

    def test_remote_consistent(self):
        self.assertEqual(self._remote_status("b", "b2", "b2").verdict,
                         "consistent")

    def test_remote_mismatch(self):
        self.assertEqual(self._remote_status("b", "b2", "b1").verdict,
                         "mismatch")

    def test_remote_unreachable(self):
        def probe(alias, path):
            raise pin_status.PinStatusError("ssh: connection refused")
        status = self._remote_status("b", "b2", "b2", probe=probe)
        self.assertEqual(status.verdict, "unreachable")
        self.assertFalse(status.reachable)

    def _report_with(self, trees) -> pin_status.PinStatusReport:
        return pin_status.build_report(
            _paths_for(self.local_root), trees, probe=self._probe())

    def test_aggregate_converged(self):
        from bigcherry.config import Tree
        b = _fake_tree(self.tmp, "b", self.upstream, "b2", "b2")
        trees = (Tree(name="brutus", alias="brutus", path=str(b),
                      required=True, role="campaign",
                      expected_tooling_revision=""),)
        report = self._report_with(trees)
        self.assertIs(report.converged, True)
        self.assertEqual(report.remotes[0].verdict, "consistent")

    def test_aggregate_diverges(self):
        from bigcherry.config import Tree
        b = _fake_tree(self.tmp, "b", self.upstream, "b1", "b1")
        trees = (Tree(name="brutus", alias="brutus", path=str(b),
                      required=True, role="campaign",
                      expected_tooling_revision=""),)
        report = self._report_with(trees)
        self.assertIs(report.converged, False)

    def test_complete_pass(self):
        from bigcherry.config import Tree
        b = _fake_tree(self.tmp, "b", self.upstream, "b2", "b2")
        trees = (Tree(name="brutus", alias="brutus", path=str(b),
                      required=True, role="campaign",
                      expected_tooling_revision=""),)
        report = self._report_with(trees)
        self.assertEqual(pin_status.complete_failures(report, trees), [])

    def test_complete_fails_on_unreachable_required(self):
        from bigcherry.config import Tree

        def probe(alias, path):
            raise pin_status.PinStatusError("refused")

        trees = (Tree(name="brutus", alias="brutus", path="/nowhere",
                      required=True, role="campaign",
                      expected_tooling_revision=""),)
        report = pin_status.build_report(
            _paths_for(self.local_root), trees, probe=probe)
        failures = pin_status.complete_failures(report, trees)
        self.assertTrue(any("unreachable" in f for f in failures))


class PolicyTierTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.upstream = _upstream(self.tmp)

    def tearDown(self):
        self._tmp.cleanup()

    def test_strict_passes_mid_rebase(self):
        root = _fake_tree(self.tmp, "t", self.upstream, "b2", "b1",
                          records={"b1": "audited"})
        _write_marker(root, "b1", "b2", commit=True)
        status = _status(root)
        report = pin_status.PinStatusReport(local=status, remotes=[],
                                            converged=True)
        self.assertEqual(status.verdict, "mid-rebase")
        self.assertEqual(pin_status.strict_failure(report), [])

    def test_strict_fails_drift(self):
        root = _fake_tree(self.tmp, "t", self.upstream, "b2", "b1",
                          records={"b1": "audited"})
        status = _status(root)
        report = pin_status.PinStatusReport(local=status, remotes=[],
                                            converged=True)
        self.assertEqual(status.verdict, "drift")
        failures = pin_status.strict_failure(report)
        self.assertEqual(len(failures), 1)
        self.assertTrue(failures[0].startswith("drift"))

    def test_strict_fails_uncommitted(self):
        root = _fake_tree(self.tmp, "t", self.upstream, "b2", "b1",
                          records={"b1": "audited"})
        _write_marker(root, "b1", "b2", commit=False)
        status = _status(root)
        report = pin_status.PinStatusReport(local=status, remotes=[],
                                            converged=True)
        self.assertEqual(status.verdict, "uncommitted-transition")
        self.assertEqual(len(pin_status.strict_failure(report)), 1)


class RepinAndPullGuardTests(unittest.TestCase):
    """(e)+(f): repin writes the marker atomically with the pin move; pull
    refuses while the marker is uncommitted.

    cmd_repin/cmd_pull read module-level bigcherry.paths / bigcherry.recipes
    constants, so the tests monkeypatch those to the fake tree and restore
    them afterwards. repin deliberately does NOT check out the vendor (the
    checkout happens on the next `pull`), so after repin the tree is
    exactly: pin=b2, vendor=b1, marker uncommitted -> uncommitted-transition.
    """

    def setUp(self):
        import bigcherry.paths as bigcherry_paths
        import bigcherry.recipes as bigcherry_recipes
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.upstream = _upstream(self.tmp)
        self.root = _fake_tree(self.tmp, "t", self.upstream, "b1", "b1",
                               records={"b1": "audited"})
        self.vendor = self.root / "vendor" / "llama.cpp"
        self._paths_mod = bigcherry_paths
        self._recipes_mod = bigcherry_recipes
        self._saved = (
            bigcherry_paths.REPO_ROOT,
            bigcherry_paths.RECIPES,
            bigcherry_recipes.RECIPES_PATH,
            os.environ.get("BIGCHERRY_LLAMA_ROOT"),
        )
        bigcherry_paths.REPO_ROOT = self.root
        bigcherry_paths.RECIPES = self.root / "config" / "recipes.toml"
        bigcherry_recipes.RECIPES_PATH = self.root / "config" / "recipes.toml"
        os.environ["BIGCHERRY_LLAMA_ROOT"] = str(self.vendor)

    def tearDown(self):
        self._paths_mod.REPO_ROOT = self._saved[0]
        self._paths_mod.RECIPES = self._saved[1]
        self._recipes_mod.RECIPES_PATH = self._saved[2]
        saved_env = self._saved[3]
        if saved_env is None:
            os.environ.pop("BIGCHERRY_LLAMA_ROOT", None)
        else:
            os.environ["BIGCHERRY_LLAMA_ROOT"] = saved_env
        self._tmp.cleanup()

    def test_repin_writes_marker_with_pin(self):
        from argparse import Namespace

        from bigcherry import __main__ as bigcherry_main
        rc = bigcherry_main.cmd_repin(Namespace(ref="b2"))
        self.assertEqual(rc, 0, "cmd_repin failed")
        pin = _PIN_LINE.search(
            (self.root / "config" / "recipes.toml").read_text(encoding="utf-8"))
        if pin is None:
            self.fail("pinned line missing after repin")
        self.assertEqual(pin.group(1), "b2")
        # repin moves the pin, NOT the checkout: vendor stays at b1
        # (detached HEAD at the b1 tag)
        self.assertEqual(_git(self.vendor, "rev-parse", "HEAD"),
                         _git(self.vendor, "rev-parse", "b1^{commit}"))
        marker_path = self.root / "releases" / "pin-transition.json"
        self.assertTrue(marker_path.is_file())
        marker = pin_transition.load(marker_path)
        if marker is None:
            self.fail("marker file missing after repin")
        self.assertEqual(marker.tag, "b2")
        self.assertEqual(marker.to_sha,
                         _git(self.vendor, "rev-parse", "b2^{commit}"))
        self.assertEqual(marker.from_sha,
                         _git(self.vendor, "rev-parse", "b1^{commit}"))
        # the declared state is exactly uncommitted-transition
        self.assertEqual(_status(self.root).verdict,
                         "uncommitted-transition")

    def test_repin_emits_advisory_ancestry_report_for_target(self):
        """RD95 wiring: a successful repin invokes the advisory ancestry
        report for the target pin."""
        from argparse import Namespace
        from unittest import mock

        import bigcherry.sources as bigcherry_sources
        from bigcherry import __main__ as bigcherry_main

        with mock.patch.object(
            bigcherry_sources, "baseline_candidates_at_pin",
            return_value={"candidate_pin": "b2", "pin_resolvable": False,
                          "candidates": []},
        ) as spy_report, mock.patch.object(
            bigcherry_sources, "print_baseline_candidates",
        ) as spy_print:
            rc = bigcherry_main.cmd_repin(Namespace(ref="b2"))
        self.assertEqual(rc, 0)
        spy_report.assert_called_once_with("b2")
        spy_print.assert_called_once()

    def test_repin_succeeds_even_if_ancestry_report_raises(self):
        """RD95 wiring: a report failure is advisory-only -- it must not fail
        the repin or lose the RE48 transition marker."""
        from argparse import Namespace
        from contextlib import redirect_stderr
        from io import StringIO
        from unittest import mock

        import bigcherry.sources as bigcherry_sources
        from bigcherry import __main__ as bigcherry_main

        with mock.patch.object(
            bigcherry_sources, "baseline_candidates_at_pin",
            side_effect=RuntimeError("boom"),
        ):
            buf = StringIO()
            with redirect_stderr(buf):
                rc = bigcherry_main.cmd_repin(Namespace(ref="b2"))
        self.assertEqual(rc, 0, "a report failure must not fail the repin")
        self.assertTrue(
            (self.root / "releases" / "pin-transition.json").is_file(),
            "the transition marker must still be written",
        )
        self.assertIn("report unavailable", buf.getvalue())

    def test_pull_refused_while_marker_uncommitted(self):
        from argparse import Namespace
        from contextlib import redirect_stderr
        import io

        from bigcherry import __main__ as bigcherry_main

        _write_marker(self.root, "b1", "b2", commit=False)
        buffer = io.StringIO()
        with redirect_stderr(buffer):
            rc = bigcherry_main.cmd_pull(
                Namespace(llama_root=None, ref=None))
        self.assertEqual(rc, 2)
        self.assertIn("uncommitted", buffer.getvalue())

    def test_pull_allowed_when_marker_absent_or_committed(self):
        from argparse import Namespace
        from contextlib import redirect_stderr
        import io

        from bigcherry import __main__ as bigcherry_main

        # committed marker: no refusal at the guard (later pull steps fail
        # on the fake tree, but NOT with the uncommitted-marker message)
        _write_marker(self.root, "b1", "b2", commit=True)
        buffer = io.StringIO()
        with redirect_stderr(buffer):
            try:
                rc = bigcherry_main.cmd_pull(
                    Namespace(llama_root=None, ref=None))
            except Exception:
                rc = "raised"
        output = buffer.getvalue()
        self.assertNotIn("pin-transition marker) is uncommitted", output)
        del rc


if __name__ == "__main__":
    unittest.main()
