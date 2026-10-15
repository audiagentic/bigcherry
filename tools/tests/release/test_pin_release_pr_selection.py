"""pin-release merges the release PR of its own component, not another component's with the same version."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bigcherry.release import pin_release  # noqa: E402

_REPO = Path(__file__).resolve().parents[3]


def _prs(*rows):
    return SimpleNamespace(returncode=0, stdout=json.dumps([
        {"number": n, "title": title, "headRefName": head, "url": f"u/{n}"} for n, title, head in rows]), stderr="")


class ReleasePrSelection(unittest.TestCase):
    def setUp(self):
        self.plan = SimpleNamespace(repo_root=_REPO, main="main", version="11552.0.0", package="engines/llamacpp")

    def test_other_component_with_the_same_version_is_not_taken(self):
        listing = _prs((106, "release:  bc-platform 11552.0.0", "release-please--branches--main--components--bc-platform"))
        with mock.patch.object(pin_release, "_run", return_value=listing):
            self.assertIsNone(pin_release._release_pr(self.plan))

    def test_own_component_is_taken(self):
        listing = _prs((106, "release:  bc-platform 11552.0.0", "release-please--branches--main--components--bc-platform"),
                       (168, "release:  bc-llamacpp 11552.0.0", "release-please--branches--main--components--bc-llamacpp"))
        with mock.patch.object(pin_release, "_run", return_value=listing):
            self.assertEqual(pin_release._release_pr(self.plan)["number"], 168)

    def test_own_component_at_another_version_is_not_taken(self):
        listing = _prs((168, "release:  bc-llamacpp 11474.9.1", "release-please--branches--main--components--bc-llamacpp"))
        with mock.patch.object(pin_release, "_run", return_value=listing):
            self.assertIsNone(pin_release._release_pr(self.plan))

    def test_release_pr_with_unexpected_files_is_refused(self):
        changed = SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"files": [
                {"path": ".release-please-manifest.json"},
                {"path": "engines/llamacpp/CHANGELOG.md"},
                {"path": "docs/releases/PLATFORM_CHANGELOG.md"},
            ]}),
            stderr="",
        )
        pr = {"number": 168, "url": "u/168"}
        with mock.patch.object(pin_release, "_run", return_value=changed):
            with self.assertRaisesRegex(pin_release.PinReleaseError, "unexpected files"):
                pin_release._verify_release_pr_files(self.plan, pr)


if __name__ == "__main__":
    unittest.main()
