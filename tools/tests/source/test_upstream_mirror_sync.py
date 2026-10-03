from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.source import upstream  # noqa: E402


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(cwd), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


class SyncMirrorRefTests(unittest.TestCase):
    def setUp(self) -> None:
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        root = Path(scratch.name)
        self.origin = root / "origin"
        self.origin.mkdir()
        _git(self.origin, "init", "-q")
        _git(self.origin, "-c", "user.email=t@t", "-c", "user.name=t",
             "commit", "-q", "--allow-empty", "-m", "base")
        _git(self.origin, "tag", "b100")
        self.mirror = root / "mirror.git"
        subprocess.run(["git", "clone", "-q", "--bare", str(self.origin), str(self.mirror)],
                       check=True, capture_output=True)
        _git(self.origin, "-c", "user.email=t@t", "-c", "user.name=t",
             "commit", "-q", "--allow-empty", "-m", "next")
        _git(self.origin, "tag", "b101")
        self.b101 = _git(self.origin, "rev-parse", "b101")

    def _resolves(self, ref: str) -> bool:
        return subprocess.run(
            ["git", "-C", str(self.mirror), "rev-parse", "--verify", "-q", f"{ref}^{{commit}}"],
            capture_output=True,
        ).returncode == 0

    def test_fetches_a_new_release_tag_into_the_mirror(self):
        self.assertFalse(self._resolves("b101"))
        self.assertIsNone(upstream.sync_mirror_ref(self.mirror, "b101", self.b101))
        self.assertTrue(self._resolves("b101"))

    def test_already_present_ref_is_a_no_op(self):
        self.assertIsNone(upstream.sync_mirror_ref(self.mirror, "b100", None))

    def test_missing_mirror_is_skipped(self):
        self.assertIsNone(upstream.sync_mirror_ref(self.mirror.parent / "absent.git", "b101", None))

    def test_unresolvable_ref_returns_a_warning(self):
        warning = upstream.sync_mirror_ref(self.mirror, "b999", None)
        self.assertIsNotNone(warning)
        self.assertIn("b999", warning)


if __name__ == "__main__":
    unittest.main()
