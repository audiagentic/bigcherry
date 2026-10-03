"""Offline mechanics tests for 1302_cuda_graph_oom_evict."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bigcherry.patcher import apply_all  # noqa: E402

_REPO = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location("patch_1302", _REPO / "patches/1302_cuda_graph_oom_evict/patch.py")
assert _spec is not None and _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

# The two upstream instantiate sites (c061df198), reduced to their surrounding shape.
_SOURCE = """\
static void ggml_cuda_graph_update_executable(ggml_backend_cuda_context * cuda_ctx, const void * graph_key) {
    ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);
    if (stat == cudaErrorGraphExecUpdateFailure) {
        (void)cudaGetLastError();
        CUDA_CHECK(cudaGraphExecDestroy(graph->instance));
        graph->instance = nullptr;
        CUDA_CHECK(cudaGraphInstantiate(&graph->instance, graph->graph, NULL, NULL, 0));
    }
}

static void evaluate(ggml_backend_cuda_context * cuda_ctx, const void * graph_key) {
    ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);
    if (graph->instance == nullptr) {
        CUDA_CHECK(cudaGraphInstantiate(&graph->instance, graph->graph, NULL, NULL, 0));
    }
}
"""


class Patch1302Mechanics(unittest.TestCase):
    def _tree(self, text=_SOURCE):
        td = tempfile.TemporaryDirectory()
        root = Path(td.name)
        path = root / "ggml/src/ggml-cuda/ggml-cuda.cu"
        path.parent.mkdir(parents=True)
        path.write_text(text, encoding="utf-8")
        return td, root, path

    def test_apply_routes_both_sites_and_is_idempotent(self):
        td, root, path = self._tree()
        with td:
            results = apply_all(_module.PATCHES, root)
            self.assertTrue(all(r.ok for r in results), [e.detail for r in results for e in r.failed])
            out = path.read_text(encoding="utf-8")
            self.assertEqual(2, out.count("bigcherry_cuda_graph_instantiate(cuda_ctx, graph);"))
            self.assertNotIn("CUDA_CHECK(cudaGraphInstantiate(", out)
            self.assertLess(out.index("static void bigcherry_cuda_graph_instantiate("),
                            out.index("static void ggml_cuda_graph_update_executable("))
            self.assertIn("err == cudaErrorMemoryAllocation", out)
            self.assertIn("BIGCHERRY_PATCH_HIT patch=1302_graph_oom_evict", out)
            # The format string's newline must stay a C escape, not a raw newline inside the literal.
            self.assertIn('evicted=%zu\\n", cuda_ctx->device, evicted);', out)

            second = apply_all(_module.PATCHES, root)
            self.assertTrue(all(r.ok for r in second), [e.detail for r in second for e in r.failed])
            self.assertEqual(out, path.read_text(encoding="utf-8"))

    def test_missing_site_fails_closed_without_writes(self):
        broken = _SOURCE.replace("        CUDA_CHECK(cudaGraphInstantiate(&graph->instance, graph->graph, NULL, NULL, 0));\n    }\n}\n",
                                 "    }\n}\n", 1)
        self.assertNotEqual(_SOURCE, broken)
        td, root, path = self._tree(broken)
        with td:
            results = apply_all(_module.PATCHES, root)
            self.assertFalse(all(r.ok for r in results))
            self.assertTrue(any(e.edit_id == "graph-oom-sites" for r in results for e in r.failed))
            self.assertEqual(broken, path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
