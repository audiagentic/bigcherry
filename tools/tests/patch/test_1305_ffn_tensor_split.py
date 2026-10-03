"""Offline mechanics tests for 1305_ffn_tensor_split (pinned src/llama-model.cpp + 1303)."""

from __future__ import annotations

import importlib.util
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bigcherry.patcher import apply_all  # noqa: E402

_REPO = Path(__file__).resolve().parents[3]
_VENDOR = _REPO / "vendor/llama.cpp/src/llama-model.cpp"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_module = _load("patch_1305", _REPO / "patches/1305_ffn_tensor_split/patch.py")
_p1303 = _load("patch_1303", _REPO / "patches/1303_attn_kv_tensor_split/patch.py")


@unittest.skipUnless(_VENDOR.exists(), "pinned vendor checkout not present")
class Patch1305Mechanics(unittest.TestCase):
    def _tree(self, with_1303=True):
        td = tempfile.TemporaryDirectory()
        root = Path(td.name)
        path = root / "src/llama-model.cpp"
        path.parent.mkdir(parents=True)
        shutil.copy2(_VENDOR, path)
        if with_1303:
            prereq = apply_all(_p1303.PATCHES, root)
            assert all(r.ok for r in prereq), [e.detail for r in prereq for e in r.failed]
        return td, root, path

    def test_apply_selects_ffn_vector_and_is_idempotent(self):
        td, root, path = self._tree()
        with td:
            results = apply_all(_module.PATCHES, root)
            self.assertTrue(all(r.ok for r in results), [e.detail for r in results for e in r.failed])
            out = path.read_text(encoding="utf-8")
            self.assertIn('std::getenv("BIGCHERRY_FFN_TS")', out)
            self.assertIn("std::regex_match(tensor_name, pattern_ffn_down_shexp_weight)", out)
            self.assertIn("!bigcherry_ffn_split.empty() && !bigcherry_use_attn_split", out)
            self.assertIn(": bigcherry_use_ffn_split ? bigcherry_ffn_split.data() : ud->model->tensor_split();", out)
            self.assertIn('ffn_ts=%s\\n"', out)
            # FFN classification is complete before the rotation and before the vector is chosen.
            self.assertLess(out.index("const bool bigcherry_use_ffn_split ="), out.index("const size_t split_rotation ="))
            self.assertLess(out.index("const bool bigcherry_use_ffn_split ="), out.index(": bigcherry_use_ffn_split ?"))

            second = apply_all(_module.PATCHES, root)
            self.assertTrue(all(r.ok for r in second), [e.detail for r in second for e in r.failed])
            self.assertEqual(out, path.read_text(encoding="utf-8"))

    def test_without_1303_fails_closed(self):
        td, root, path = self._tree(with_1303=False)
        with td:
            before = path.read_text(encoding="utf-8")
            results = apply_all(_module.PATCHES, root)
            self.assertFalse(all(r.ok for r in results))
            self.assertEqual(before, path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
