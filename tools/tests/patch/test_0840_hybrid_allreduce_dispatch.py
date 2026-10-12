"""Offline mechanics for 0840's adaptive-provider edit on 0860's owned provider header."""

from __future__ import annotations

import importlib.util
import re
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bigcherry.core import paths  # noqa: E402
from bigcherry.patch.pinned_source import production_source  # noqa: E402
from bigcherry.patcher import apply_all  # noqa: E402

_REPO = Path(__file__).resolve().parents[3]
_ID = "0840_hybrid_allreduce_dispatch"
_PROVIDER = "ggml/src/ggml-cuda/bc-allreduce-provider-cli.cuh"


def _load():
    spec = importlib.util.spec_from_file_location(
        "patch_0840", _REPO / "engines/llamacpp/patches" / _ID / "patch.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_P = _load()


@unittest.skipUnless(paths.llama_root().is_dir(), "pinned vendor checkout not present")
class Patch0840OwnedProviderAnchor(unittest.TestCase):
    @staticmethod
    def _patches():
        return tuple(fp for fp in _P.PATCHES if fp.path == _PROVIDER)

    def _root(self, td):
        return production_source((_PROVIDER,), td, stop_before=_ID)

    def test_owned_provider_edit_guards_and_idempotence(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._root(td)
            path = root / _PROVIDER
            before = path.read_text(encoding="utf-8")
            patches = self._patches()
            self.assertEqual(len(patches), 1)
            self.assertIn('    if (p == "adaptive" || p == "p2p" || p == "root3") {\n', before)
            for edit in patches[0].edits:
                self.assertIsNone(re.search(edit.guard_pattern(), before, re.MULTILINE), edit.id)

            result = apply_all(patches, root)
            self.assertTrue(all(r.ok for r in result), [e.detail for r in result for e in r.failed])
            after = path.read_text(encoding="utf-8")
            self.assertIn('    if (p == "p2p" || p == "root3") {\n', after)
            self.assertNotIn('    if (p == "adaptive" || p == "p2p" || p == "root3") {\n', after)
            for edit in patches[0].edits:
                self.assertIsNotNone(re.search(edit.guard_pattern(), after, re.MULTILINE), edit.id)

            second = apply_all(patches, root)
            self.assertTrue(all(r.ok for r in second), [e.detail for r in second for e in r.failed])
            self.assertEqual(after, path.read_text(encoding="utf-8"))

    def test_missing_owned_provider_anchor_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._root(td)
            path = root / _PROVIDER
            before = path.read_text(encoding="utf-8")
            anchor = '    if (p == "adaptive" || p == "p2p" || p == "root3") {\n'
            self.assertEqual(before.count(anchor), 1)
            drifted = before.replace(anchor, '    if (p == "adaptive" || p == "p2p") {\n', 1)
            path.write_text(drifted, encoding="utf-8")
            result = apply_all(self._patches(), root)
            self.assertFalse(all(r.ok for r in result))
            self.assertEqual(drifted, path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
