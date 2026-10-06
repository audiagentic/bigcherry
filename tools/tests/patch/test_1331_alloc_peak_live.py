"""Offline mechanics tests for 1331_alloc_peak_live (pinned ggml-alloc.c, alone and with 1329)."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bigcherry.patcher import apply_all  # noqa: E402
from bigcherry.patch.pinned_source import copy_pinned  # noqa: E402

_REPO = Path(__file__).resolve().parents[3]
_REL = "ggml/src/ggml-alloc.c"
_VENDOR = _REPO / "vendor/llama.cpp" / _REL


def _load(pid: str):
    spec = importlib.util.spec_from_file_location("patch_" + pid, _REPO / "patches" / pid / "patch.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_P = _load("1331_alloc_peak_live")


@unittest.skipUnless(_VENDOR.exists(), "pinned vendor checkout not present")
class Patch1331Mechanics(unittest.TestCase):
    def _tree(self, td):
        root = Path(td)
        (root / _REL).parent.mkdir(parents=True)
        copy_pinned(_VENDOR, root / _REL)
        return root

    def _alloc_patches(self, mod):
        return [p for p in mod.PATCHES if p.path == _REL]

    def test_apply_and_idempotent(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._tree(td)
            res = apply_all(self._alloc_patches(_P), root)
            self.assertTrue(all(r.ok for r in res), [e.detail for r in res for e in r.failed])
            out = (root / _REL).read_text(encoding="utf-8")
            self.assertLess(out.index("static void bc_peak_report"), out.index("static void ggml_gallocr_allocate_node(ggml_gallocr_t"))
            alloc_fn = out[out.index("static void ggml_gallocr_allocate_node("):out.index("static void ggml_gallocr_free_node(")]
            self.assertIn("bc_peak_alloc(galloc, buffer_id, node, size);", alloc_fn)
            free_fn = out[out.index("static void ggml_gallocr_free_node("):out.index("static int get_node_buffer_id")]
            self.assertIn("bc_peak_free(buffer_id, size);", free_fn)
            impl = out[out.index("static void ggml_gallocr_alloc_graph_impl"):out.index("static bool ggml_gallocr_reserve_n_impl")]
            self.assertLess(impl.index("bc_peak_begin();"), impl.index("// allocate leafs"))
            self.assertTrue(impl.rstrip().endswith("bc_peak_report(graph);  // bigcherry 1331\n}"))
            second = apply_all(self._alloc_patches(_P), root)
            self.assertTrue(all(r.ok for r in second))
            self.assertEqual(out, (root / _REL).read_text(encoding="utf-8"))

    def test_composes_with_1329(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._tree(td)
            for mod in (_load("1329_alloc_top_trace"), _P):
                res = apply_all(self._alloc_patches(mod), root)
                self.assertTrue(all(r.ok for r in res), [e.detail for r in res for e in r.failed])


if __name__ == "__main__":
    unittest.main()
