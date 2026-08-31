"""cmd_pull's --source ref resolution (compat.recipe removal plan).

gpt-dev-agent-reviewed design, dev-gpt-agent gateway session
ses_5307d9c58ec645cb: "pull needs the source REF, not build/platform --
resolve the v2 source name, read ref from cfg.sources[source] directly."

Exercises only the ref-resolution branch in isolation (mocking every git/
network side effect) -- the git-mutating behavior itself is already
covered by RepinAndPullGuardTests in test_pin_status.py, which this test
does not duplicate.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.cli import source as cli_source  # noqa: E402


_RECIPES_TOML = """\
version = 2
pinned = "b10705"

[patch-set.framework]
patches = []
required-state = "validated"

[source.bigcherry]
ref = "pinned"
overlay = true
patch-sets = ["framework"]

[source.pinned-elsewhere]
ref = "b9999"
overlay = false
patch-sets = []

[compat.recipe.bigcherry]
ref = "pinned"
states = ["validated"]
builds = ["record"]

[build.record]
"""


class PullSourceRefResolutionTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="bigcherry-pull-source-test-")
        self.addCleanup(self._tmp.cleanup)
        self.recipes_path = Path(self._tmp.name) / "recipes.toml"
        self.recipes_path.write_text(_RECIPES_TOML, encoding="utf-8")

    def _args(self, **kwargs):
        base = {"llama_root": None, "ref": None, "recipe": None, "source": None,
                "full": False}
        base.update(kwargs)
        return Namespace(**base)

    def test_source_with_pinned_ref_resolves_to_cfg_pinned(self):
        from bigcherry.core import paths as core_paths

        resolved_refs = []

        def fake_resolve_ref(ref):
            resolved_refs.append(ref)
            return ref

        with mock.patch.object(core_paths, "RECIPES", self.recipes_path), \
                mock.patch("bigcherry.__main__._uncommitted_pin_change", return_value=None), \
                mock.patch("bigcherry.__main__.pin_transition.committed_state"), \
                mock.patch("bigcherry.__main__.paths.REPO_ROOT", Path(self._tmp.name)), \
                mock.patch("bigcherry.__main__.upstream.resolve_ref", side_effect=fake_resolve_ref), \
                mock.patch("bigcherry.__main__.upstream.clear_stale_locks", return_value=[]), \
                mock.patch("bigcherry.__main__.upstream.ensure_ref", return_value="b10705"), \
                mock.patch("bigcherry.__main__._run"), \
                mock.patch("bigcherry.__main__._record_for") as record_for, \
                mock.patch.object(Path, "exists", return_value=True):
            record_for.return_value.revision = "deadbeef"
            record_for.return_value.release_tag = "b10705"
            cli_source.cmd_pull(self._args(source="bigcherry"))

        # source.bigcherry has ref="pinned" -> must resolve through
        # cfg.pinned ("b10705"), the same convention campaign/source.py
        # itself uses, not the literal string "pinned".
        self.assertEqual(resolved_refs, ["b10705"])

    def test_source_with_explicit_ref_uses_that_ref_not_the_pin(self):
        from bigcherry.core import paths as core_paths

        resolved_refs = []

        def fake_resolve_ref(ref):
            resolved_refs.append(ref)
            return ref

        with mock.patch.object(core_paths, "RECIPES", self.recipes_path), \
                mock.patch("bigcherry.__main__._uncommitted_pin_change", return_value=None), \
                mock.patch("bigcherry.__main__.pin_transition.committed_state"), \
                mock.patch("bigcherry.__main__.paths.REPO_ROOT", Path(self._tmp.name)), \
                mock.patch("bigcherry.__main__.upstream.resolve_ref", side_effect=fake_resolve_ref), \
                mock.patch("bigcherry.__main__.upstream.clear_stale_locks", return_value=[]), \
                mock.patch("bigcherry.__main__.upstream.ensure_ref", return_value="b9999"), \
                mock.patch("bigcherry.__main__._run"), \
                mock.patch("bigcherry.__main__._record_for") as record_for, \
                mock.patch.object(Path, "exists", return_value=True):
            record_for.return_value.revision = "deadbeef"
            record_for.return_value.release_tag = "b9999"
            cli_source.cmd_pull(self._args(source="pinned-elsewhere"))

        self.assertEqual(resolved_refs, ["b9999"])

    def test_unknown_source_fails_closed(self):
        from bigcherry.core import paths as core_paths

        with mock.patch.object(core_paths, "RECIPES", self.recipes_path), \
                mock.patch("bigcherry.__main__._uncommitted_pin_change", return_value=None), \
                mock.patch("bigcherry.__main__.pin_transition.committed_state"), \
                mock.patch("bigcherry.__main__.paths.REPO_ROOT", Path(self._tmp.name)):
            rc = cli_source.cmd_pull(self._args(source="not-a-real-source"))
        self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
