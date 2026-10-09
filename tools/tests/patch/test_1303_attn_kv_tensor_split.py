"""Offline mechanics tests for 1303_attn_kv_tensor_split (against the pinned src/llama-model.cpp)."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bigcherry.core import paths  # noqa: E402
from bigcherry.patcher import apply_all  # noqa: E402
from bigcherry.patch.pinned_source import copy_pinned  # noqa: E402

_REPO = Path(__file__).resolve().parents[3]
_VENDOR = paths.llama_root() / "src/llama-model.cpp"
_spec = importlib.util.spec_from_file_location("patch_1303", _REPO / "engines/llamacpp/patches/1303_attn_kv_tensor_split/patch.py")
assert _spec is not None and _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


@unittest.skipUnless(_VENDOR.exists(), "pinned vendor checkout not present")
class Patch1303Mechanics(unittest.TestCase):
    def _tree(self):
        td = tempfile.TemporaryDirectory()
        root = Path(td.name)
        path = root / "src/llama-model.cpp"
        path.parent.mkdir(parents=True)
        copy_pinned(_VENDOR, path)
        return td, root, path

    def test_apply_selects_vector_and_rotation_and_is_idempotent(self):
        td, root, path = self._tree()
        with td:
            results = apply_all(_module.PATCHES, root)
            self.assertTrue(all(r.ok for r in results), [e.detail for r in results for e in r.failed])
            out = path.read_text(encoding="utf-8")
            self.assertIn('std::getenv("BIGCHERRY_ATTN_TS")', out)
            self.assertIn('std::getenv("BIGCHERRY_ATTN_ROTATE")', out)
            self.assertIn("!hparams.is_recr(tc.il)", out)
            self.assertIn("std::regex_match(tensor_name, pattern_kv_cache)", out)
            self.assertIn("? bigcherry_attn_split.split.data() : ud->model->tensor_split();", out)
            self.assertEqual(3, out.count("(j + split_rotation) % ud->n_devices"))
            self.assertNotIn("(j + tc.rotation) % ud->n_devices", out)
            # The config is parsed before use, and the selection precedes the vector choice.
            self.assertLess(out.index("struct bigcherry_attn_split_config {"), out.index("const bool bigcherry_use_attn_split ="))
            self.assertLess(out.index("const bool bigcherry_use_attn_split ="), out.index("? bigcherry_attn_split.split.data()"))
            self.assertIn('attn_rotate=%d\\n"', out)

            second = apply_all(_module.PATCHES, root)
            self.assertTrue(all(r.ok for r in second), [e.detail for r in second for e in r.failed])
            self.assertEqual(out, path.read_text(encoding="utf-8"))

    def test_rotation_site_drift_fails_closed_without_writes(self):
        td, root, path = self._tree()
        with td:
            broken = path.read_text(encoding="utf-8").replace("(j + tc.rotation) % ud->n_devices", "(j + tc.rotation + 0) % ud->n_devices", 1)
            path.write_text(broken, encoding="utf-8")
            results = apply_all(_module.PATCHES, root)
            self.assertFalse(all(r.ok for r in results))
            self.assertTrue(any(e.edit_id == "attn-ts-rotation" for r in results for e in r.failed))
            self.assertEqual(broken, path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
