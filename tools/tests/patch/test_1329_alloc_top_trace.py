"""Offline mechanics tests for 1329_alloc_top_trace (pinned ggml/src/ggml-alloc.c)."""

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
_VENDOR = _REPO / "vendor/llama.cpp/ggml/src/ggml-alloc.c"
_spec = importlib.util.spec_from_file_location("patch_1329", _REPO / "patches/1329_alloc_top_trace/patch.py")
assert _spec is not None and _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


@unittest.skipUnless(_VENDOR.exists(), "pinned vendor checkout not present")
class Patch1329Mechanics(unittest.TestCase):
    def test_apply_and_idempotent(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = root / "ggml/src/ggml-alloc.c"
            path.parent.mkdir(parents=True)
            shutil.copy2(_VENDOR, path)
            results = apply_all(_module.PATCHES, root)
            self.assertTrue(all(r.ok for r in results), [e.detail for r in results for e in r.failed])
            out = path.read_text(encoding="utf-8")
            helper = out.index("static void bc_alloc_top_trace(")
            hook = out.index("bc_alloc_top_trace(graph, node_buffer_ids);  // bigcherry 1329")
            self.assertLess(helper, hook)
            self.assertLess(out.index("bool ggml_gallocr_reserve_n(ggml_gallocr_t galloc"), hook)
            self.assertLess(hook, out.index("return ggml_gallocr_reserve_n_impl(galloc, graph, node_buffer_ids, leaf_buffer_ids, /*no_alloc =*/ false);"))
            second = apply_all(_module.PATCHES, root)
            self.assertTrue(all(r.ok for r in second), [e.detail for r in second for e in r.failed])
            self.assertEqual(out, path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
