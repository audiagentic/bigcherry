"""One release-record key for every writer and reader, and pull fetching the pinned tag.

The key is the tag the checkout sits exactly on (the pinned ref: ``b11552`` for llama.cpp, ``v1.7.0`` for
radiance), else the 12-hex commit prefix. Uses real local git repos (a bare upstream plus a clone)."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.release import pin_status  # noqa: E402
from bigcherry.release import records as releases  # noqa: E402
from bigcherry.source import upstream  # noqa: E402


def _git(cwd: Path, *args: str) -> str:
    done = subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True, text=True)
    return done.stdout.strip()


class ReleaseKeyTests(unittest.TestCase):
    def test_key_is_tag_else_twelve_hex(self):
        sha = "1e968f6e1d86e2544455b00e9e57c09690d9ea10"
        self.assertEqual(releases.release_key(sha, "v1.7.0"), "v1.7.0")
        self.assertEqual(releases.release_key(sha, "b11552"), "b11552")
        self.assertEqual(releases.release_key(sha, ""), "1e968f6e1d86")

    def test_slug_path_and_load_agree(self):
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(releases, "RELEASES_DIR", Path(d)), \
                mock.patch.object(releases, "INDEX_PATH", Path(d) / "index.json"):
            for tag in ("b11552", "v1.7.0"):
                rec = releases.ReleaseRecord(revision="a" * 40, release_tag=tag)
                self.assertEqual(rec.slug(), tag)
                self.assertEqual(rec.save(), Path(d) / f"{tag}.json")
                self.assertEqual(releases.load("a" * 40, tag).slug(), tag)


class PullFetchesPinnedTagTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        tmp = Path(self._tmp.name)
        src = tmp / "src"
        src.mkdir()
        _git(src, "init", "-q", "-b", "main")
        _git(src, "config", "user.email", "t@t")
        _git(src, "config", "user.name", "t")
        (src / "f").write_text("x")
        _git(src, "add", "f")
        _git(src, "commit", "-q", "-m", "c")
        for tag in ("v1.7.0", "b11552"):
            _git(src, "tag", tag)  # two engines' tag styles; both resolve to the one commit
        self.bare = tmp / "up.git"
        subprocess.run(["git", "clone", "-q", "--bare", str(src), str(self.bare)], check=True)
        self.clone = tmp / "vendor"
        subprocess.run(["git", "init", "-q", str(self.clone)], check=True)
        _git(self.clone, "remote", "add", "origin", str(self.bare))
        self.sha = _git(src, "rev-parse", "HEAD")

    def test_ensure_ref_lands_a_local_non_llamacpp_tag(self):
        self.assertEqual(upstream.ensure_ref(self.clone, "v1.7.0", deepen=False), "v1.7.0")
        self.assertEqual(_git(self.clone, "rev-parse", "v1.7.0^{commit}"), self.sha)
        self.assertEqual(_git(self.clone, "tag", "--list").split(), ["v1.7.0"])

    def test_ensure_ref_branch_still_uses_fetch_head(self):
        self.assertEqual(upstream.ensure_ref(self.clone, "main", deepen=False), "FETCH_HEAD")

    def test_record_round_trip_pin_bump_writes_readers_find(self):
        # writer side: pull fetched the tag, so record_for_checkout keys the record by the pinned tag
        upstream.ensure_ref(self.clone, "v1.7.0", deepen=False)
        _git(self.clone, "checkout", "-q", "--force", "v1.7.0")
        releases_dir = Path(self._tmp.name) / "releases"
        with mock.patch.object(releases, "RELEASES_DIR", releases_dir), \
                mock.patch.object(releases, "INDEX_PATH", releases_dir / "index.json"):
            record = releases.record_for_checkout(self.clone)
            self.assertEqual(record.slug(), "v1.7.0")
            record.audit = {"passed": True}
            record.save()
            # readers: pin-release/release-notes open releases/<pinned ref>.json; pin-status maps revision->stage
            record_path = releases_dir / "v1.7.0.json"
            self.assertTrue(record_path.is_file())
            self.assertEqual(json.loads(record_path.read_text(encoding="utf-8"))["revision"], self.sha)
            self.assertEqual(pin_status._release_records(releases_dir), {self.sha: "pulled"})
            index = json.loads((releases_dir / "index.json").read_text(encoding="utf-8"))
            self.assertEqual([r["slug"] for r in index["releases"]], ["v1.7.0"])
            releases.validate_index_consistency()


if __name__ == "__main__":
    unittest.main()
