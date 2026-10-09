"""Offline mechanics tests for 1304_cuda_graph_lru_cap (pinned common.cuh + 1302-patched ggml-cuda.cu shape)."""

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
_COMMON = paths.llama_root() / "ggml/src/ggml-cuda/common.cuh"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_module = _load("patch_1304", _REPO / "engines/llamacpp/patches/1304_cuda_graph_lru_cap/patch.py")
_p1302 = _load("patch_1302", _REPO / "engines/llamacpp/patches/1302_cuda_graph_oom_evict/patch.py")

_GGML_CUDA = """\
static void ggml_cuda_graph_update_executable(ggml_backend_cuda_context * cuda_ctx, const void * graph_key) {
    ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);
    CUDA_CHECK(cudaGraphInstantiate(&graph->instance, graph->graph, NULL, NULL, 0));
}
static void evaluate(ggml_backend_cuda_context * cuda_ctx, ggml_cuda_graph * graph) {
    CUDA_CHECK(cudaGraphInstantiate(&graph->instance, graph->graph, NULL, NULL, 0));
}
"""


@unittest.skipUnless(_COMMON.exists(), "pinned vendor checkout not present")
class Patch1304Mechanics(unittest.TestCase):
    def _tree(self):
        td = tempfile.TemporaryDirectory()
        root = Path(td.name)
        cuda = root / "ggml/src/ggml-cuda"
        cuda.mkdir(parents=True)
        copy_pinned(_COMMON, cuda / "common.cuh")
        (cuda / "ggml-cuda.cu").write_text(_GGML_CUDA, encoding="utf-8")
        prereq = apply_all(_p1302.PATCHES, root)
        assert all(r.ok for r in prereq), [e.detail for r in prereq for e in r.failed]
        return td, root, cuda

    def test_apply_and_idempotent(self):
        td, root, cuda = self._tree()
        with td:
            results = apply_all(_module.PATCHES, root)
            self.assertTrue(all(r.ok for r in results), [e.detail for r in results for e in r.failed])
            common = (cuda / "common.cuh").read_text(encoding="utf-8")
            cu = (cuda / "ggml-cuda.cu").read_text(encoding="utf-8")
            self.assertIn('getenv("BIGCHERRY_CUDA_GRAPH_CAP")', common)
            self.assertIn("uint64_t bigcherry_graph_evictions = 0;", common)
            self.assertIn("capture_status == cudaStreamCaptureStatusNone", common)
            # hip.h does not map cudaStreamIsCapturing; the HIP branch must call hipStreamIsCapturing itself.
            self.assertIn("hipStreamIsCapturing(stream(), &capture_status) == hipSuccess", common)
            self.assertIn("cuda_graphs.erase(lru);", common)
            self.assertIn('evictions=%llu\\n"', common)
            self.assertIn('getenv("BIGCHERRY_GRAPH_MEMLOG")', cu)
            self.assertIn('cached=%zu\\n"', cu)
            # The LRU check sits before the emplace of the new key.
            self.assertLess(common.index("cuda_graphs.erase(lru);"),
                            common.index("it = cuda_graphs.emplace(first_node_ptr"))

            second = apply_all(_module.PATCHES, root)
            self.assertTrue(all(r.ok for r in second), [e.detail for r in second for e in r.failed])
            self.assertEqual(common, (cuda / "common.cuh").read_text(encoding="utf-8"))
            self.assertEqual(cu, (cuda / "ggml-cuda.cu").read_text(encoding="utf-8"))

    def test_without_1302_fails_closed(self):
        td = tempfile.TemporaryDirectory()
        with td:
            root = Path(td.name)
            cuda = root / "ggml/src/ggml-cuda"
            cuda.mkdir(parents=True)
            copy_pinned(_COMMON, cuda / "common.cuh")
            (cuda / "ggml-cuda.cu").write_text(_GGML_CUDA, encoding="utf-8")
            results = apply_all(_module.PATCHES, root)
            self.assertFalse(all(r.ok for r in results))
            self.assertTrue(any(e.edit_id == "graph-memlog" for r in results for e in r.failed))
            self.assertEqual(_GGML_CUDA, (cuda / "ggml-cuda.cu").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
