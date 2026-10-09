"""Offline mechanics tests for 1007_meta_subgraph_realloc_fix."""

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
_V = _REPO / "vendor/llama.cpp"
_F = "ggml/src/ggml-backend-meta.cpp"


def _load():
    spec = importlib.util.spec_from_file_location("patch_1007", _REPO / "engines/llamacpp/patches/1007_meta_subgraph_realloc_fix/patch.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_P = _load()


@unittest.skipUnless((_V / _F).exists(), "pinned vendor checkout not present")
class Patch1007Mechanics(unittest.TestCase):
    def _root(self, td):
        root = Path(td)
        (root / _F).parent.mkdir(parents=True, exist_ok=True)
        copy_pinned(_V / _F, root / _F)
        return root

    def test_apply_and_idempotent(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._root(td)
            res = apply_all(_P.PATCHES, root)
            self.assertTrue(all(r.ok for r in res), [e.detail for r in res for e in r.failed])
            src = (root / _F).read_text(encoding="utf-8")
            self.assertIn("for (size_t i = 0; i < backend_ctx->max_subgraphs; i++) {", src)
            self.assertIn("ggml_new_graph_custom(backend_ctx->ctx.get(), backend_ctx->max_nnodes, /*grads =*/ false);", src)
            self.assertNotIn("ggml_new_graph_custom(backend_ctx->ctx.get(), cgraph->n_nodes, /*grads =*/ false);", src)
            # the re-creation sits between the context reset and the aux graphs
            self.assertLess(src.index("backend_ctx->ctx.reset(ggml_init(params));"), src.index("BigCherry 1007"))
            self.assertLess(src.index("BigCherry 1007"), src.index("backend_ctx->cgraphs_aux.resize("))
            second = apply_all(_P.PATCHES, root)
            self.assertTrue(all(r.ok for r in second), [e.detail for r in second for e in r.failed])
            self.assertEqual(src, (root / _F).read_text(encoding="utf-8"))

    def test_missing_anchor_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._root(td)
            p = root / _F
            p.write_text(p.read_text(encoding="utf-8").replace("i < n_subgraphs; i++) {\n                    bcj.cgraphs[i].cgraph_main",
                                                               "i < n_sub; i++) {\n                    bcj.cgraphs[i].cgraph_main"), encoding="utf-8")
            res = apply_all(_P.PATCHES, root)
            self.assertFalse(all(r.ok for r in res))


if __name__ == "__main__":
    unittest.main()
