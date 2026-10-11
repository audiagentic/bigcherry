"""Offline mechanics tests for 1237_rd30_moe_mmq_compact_grid on the composed b11552 source."""

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
_ID = "1237_rd30_moe_mmq_compact_grid"


def _load():
    spec = importlib.util.spec_from_file_location("patch_1237", _REPO / "engines/llamacpp/patches" / _ID / "patch.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_P = _load()
_FILES = sorted({fp.path for fp in _P.PATCHES})


@unittest.skipUnless(paths.llama_root().is_dir(), "pinned vendor checkout not present")
class Patch1237Mechanics(unittest.TestCase):
    def test_apply_on_composition_and_idempotent(self):
        with tempfile.TemporaryDirectory() as td:
            root = production_source(_FILES, td, stop_before=_ID)
            # no guard may already match: a guard that matches an earlier patch's output skips its edit silently
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
            self.assertIn("zt = rd30_block_expert[m_block];", texts["ggml/src/ggml-cuda/mmq.cuh"])
            second = apply_all(_P.PATCHES, root)
            self.assertTrue(all(r.ok for r in second), [e.detail for r in second for e in r.failed])
            for path, text in texts.items():
                self.assertEqual(text, (root / path).read_text(encoding="utf-8"))

    def test_missing_anchor_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root = production_source(_FILES, td, stop_before=_ID)
            # the first file without the text its edits attach to
            (root / _P.PATCHES[0].path).write_text("// moved upstream" + chr(10), encoding="utf-8")
            res = apply_all(_P.PATCHES, root)
            self.assertFalse(all(r.ok for r in res))


if __name__ == "__main__":
    unittest.main()
