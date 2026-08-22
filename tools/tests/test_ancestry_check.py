"""RD95: real pin-ancestry checking (sources.ancestral_to_pin), against a
local throwaway git repo -- zero network dependency, unlike sources.py's
existing `_check()` (which clones real remote repos and is integration-
only). Closes the gap this session found by hand twice: a commit being
"merged upstream" (mainline/master's moving tip) does not mean it is
ancestral to THIS project's own pinned vendor revision (RD68a/RD68b:
merged hours after the pin was cut)."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bigcherry import sources as src  # noqa: E402


def _run(cwd: Path, *argv: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(cwd), *argv], capture_output=True, text=True, check=True)
    return result.stdout.strip()


def _commit(cwd: Path, filename: str, content: str) -> str:
    (cwd / filename).write_text(content, encoding="utf-8")
    _run(cwd, "add", filename)
    _run(cwd, "commit", "-m", f"add {filename}", "--no-gpg-sign")
    return _run(cwd, "rev-parse", "HEAD")


class _ThrowawayRepo:
    """A real local git repo, torn down after the test. Real ancestry
    relationships, not mocked git output -- this is the whole point of
    testing against actual git rather than stubbing subprocess.run."""

    def __enter__(self) -> Path:
        self._tmp = tempfile.TemporaryDirectory(prefix="bc-ancestry-test-")
        root = Path(self._tmp.name)
        _run(root, "init", "-q")
        _run(root, "config", "user.email", "test@example.invalid")
        _run(root, "config", "user.name", "Test")
        self.root = root
        return root

    def __exit__(self, *exc) -> None:
        self._tmp.cleanup()


class AncestralToPinTests(unittest.TestCase):
    def test_real_ancestor_is_reported_ancestral(self):
        with _ThrowawayRepo() as root:
            older = _commit(root, "a.txt", "1")
            newer = _commit(root, "b.txt", "2")  # older is an ancestor of newer
            _run(root, "tag", "the-pin", newer)
            verdict = src.ancestral_to_pin(older, vendor_root=root, pin_ref="the-pin")
            self.assertEqual(verdict, "ancestral")

    def test_the_rd68_case_not_yet_ancestral_at_pin_time(self):
        # Exactly the real scenario this function exists for: a commit that
        # merges into history AFTER the pin was already cut is NOT
        # ancestral to that pin, even though it's real, present, later
        # history in the same repo.
        with _ThrowawayRepo() as root:
            pinned = _commit(root, "a.txt", "1")
            _run(root, "tag", "the-pin", pinned)
            later = _commit(root, "b.txt", "2")  # merged AFTER the pin was cut
            verdict = src.ancestral_to_pin(later, vendor_root=root, pin_ref="the-pin")
            self.assertEqual(verdict, "not-ancestral")

    def test_candidate_sha_not_present_locally_is_unknown_not_ancestral(self):
        # A shallow/incomplete local checkout must never report "ancestral"
        # for a commit it cannot actually see -- unresolvable is not
        # evidence of redundancy, it is evidence of nothing.
        with _ThrowawayRepo() as root:
            pinned = _commit(root, "a.txt", "1")
            _run(root, "tag", "the-pin", pinned)
            fake_sha = "d34db33f" * 5
            verdict = src.ancestral_to_pin(fake_sha, vendor_root=root, pin_ref="the-pin")
            self.assertEqual(verdict, "unknown")

    def test_pin_ref_itself_not_present_is_unknown(self):
        with _ThrowawayRepo() as root:
            some_commit = _commit(root, "a.txt", "1")
            verdict = src.ancestral_to_pin(
                some_commit, vendor_root=root, pin_ref="no-such-tag")
            self.assertEqual(verdict, "unknown")

    def test_missing_vendor_root_is_unknown_not_an_exception(self):
        missing = Path(tempfile.gettempdir()) / "bc-ancestry-does-not-exist-xyz"
        verdict = src.ancestral_to_pin("a" * 40, vendor_root=missing, pin_ref="anything")
        self.assertEqual(verdict, "unknown")

    def test_sibling_commit_neither_ancestor_nor_descendant_is_not_ancestral(self):
        with _ThrowawayRepo() as root:
            base = _commit(root, "a.txt", "1")
            _run(root, "tag", "the-pin", base)
            _run(root, "checkout", "-q", "-b", "branch-a")
            _commit(root, "b.txt", "2")
            _run(root, "checkout", "-q", "-b", "branch-b", base)
            sibling = _commit(root, "c.txt", "3")
            verdict = src.ancestral_to_pin(sibling, vendor_root=root, pin_ref="the-pin")
            self.assertEqual(verdict, "not-ancestral")

    def test_pin_ref_defaults_to_real_project_config_when_omitted(self):
        # Real integration point: pin_ref=None must resolve through
        # recipes.load_config().pinned (this project's real config), not
        # silently no-op. Exercised against the throwaway repo (which has
        # no tag named after the real pin, e.g. "b10502"), so the real
        # config value is genuinely consulted and then fails closed to
        # "unknown" via the rev-parse-failure path -- never crashes, never
        # silently reports "ancestral".
        with _ThrowawayRepo() as root:
            commit = _commit(root, "a.txt", "1")
            verdict = src.ancestral_to_pin(commit, vendor_root=root, pin_ref=None)
            self.assertEqual(verdict, "unknown")


if __name__ == "__main__":
    unittest.main()
