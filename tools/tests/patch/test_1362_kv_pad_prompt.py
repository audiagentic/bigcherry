"""Offline mechanics tests for 1362_kv_pad_prompt on the composed b11552 source."""

from __future__ import annotations

import importlib.util
import re
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bigcherry.core import paths  # noqa: E402
from bigcherry.patcher import apply_all  # noqa: E402
from bigcherry.patch.pinned_source import production_source  # noqa: E402

_REPO = Path(__file__).resolve().parents[3]
_ID = "1362_kv_pad_prompt"


def _load():
    spec = importlib.util.spec_from_file_location("patch_1362", _REPO / "engines/llamacpp/patches" / _ID / "patch.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_P = _load()
_FILES = sorted({fp.path for fp in _P.PATCHES})


@unittest.skipUnless(paths.llama_root().is_dir(), "pinned vendor checkout not present")
class Patch1362Mechanics(unittest.TestCase):
    def test_apply_on_composition_and_idempotent(self):
        with tempfile.TemporaryDirectory() as td:
            root = production_source(_FILES, td, stop_before=_ID)

            for fp in _P.PATCHES:
                before = (root / fp.path).read_text(encoding="utf-8")
                for edit in fp.edits:
                    self.assertIsNone(re.search(edit.guard, before, re.M), (fp.path, edit.id))

            res = apply_all(_P.PATCHES, root)
            self.assertTrue(all(r.ok for r in res), [e.detail for r in res for e in r.failed])

            texts = {fp.path: (root / fp.path).read_text(encoding="utf-8") for fp in _P.PATCHES}
            for fp in _P.PATCHES:
                for edit in fp.edits:
                    self.assertIsNotNone(re.search(edit.guard, texts[fp.path], re.M), (fp.path, edit.id))

            source = texts["src/llama-kv-cache.cpp"]
            self.assertIn('getenv("BIGCHERRY_KV_PAD")', source)
            self.assertIn("BIGCHERRY_PATCH_HIT patch=1362_kv_pad_prompt pad=%u", source)
            self.assertIn("std::max(n_pad, std::max(256u, bc_kv_pad_cur))", source)

            second = apply_all(_P.PATCHES, root)
            self.assertTrue(all(r.ok for r in second), [e.detail for r in second for e in r.failed])
            for path, text in texts.items():
                self.assertEqual(text, (root / path).read_text(encoding="utf-8"))

    def test_missing_anchor_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root = production_source(_FILES, td, stop_before=_ID)
            (root / _P.PATCHES[0].path).write_text("// moved upstream\n", encoding="utf-8")
            res = apply_all(_P.PATCHES, root)
            self.assertFalse(all(r.ok for r in res))


if __name__ == "__main__":
    unittest.main()
