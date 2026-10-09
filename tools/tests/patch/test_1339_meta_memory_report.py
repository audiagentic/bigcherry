"""Offline mechanics tests for 1339_meta_memory_report (per-device memory report of the meta backend)."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bigcherry.patcher import apply_all  # noqa: E402

_REPO = Path(__file__).resolve().parents[3]
_V = _REPO / "vendor/llama.cpp"
_PIN = "HEAD"  # the vendor checkout is at the pinned revision
_META = "ggml/src/ggml-backend-meta.cpp"
_CUDA = "ggml/src/ggml-cuda/ggml-cuda.cu"


def _load():
    spec = importlib.util.spec_from_file_location("patch_1340_merged", _REPO / "engines/llamacpp/patches/1340_meta_per_device_arena/patch.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pinned(path):
    try:
        res = subprocess.run(["git", "-C", str(_V), "show", f"{_PIN}:{path}"], capture_output=True, text=True,
                             encoding="utf-8", check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return res.stdout


_P = _load()
_PATCHES = [p for p in _P.PATCHES if p.description.startswith("1339:")]
_SRC = {path: _pinned(path) for path in (_META, _CUDA)}


@unittest.skipUnless(all(text is not None for text in _SRC.values()), "pinned vendor repository not present")
class Patch1339Mechanics(unittest.TestCase):
    def _root(self, td, overrides=None):
        root = Path(td)
        for path, text in _SRC.items():
            (root / path).parent.mkdir(parents=True, exist_ok=True)
            (root / path).write_text((overrides or {}).get(path, text), encoding="utf-8", newline="\n")
        return root

    def test_apply_and_idempotent(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._root(td)
            res = apply_all(_PATCHES, root)
            self.assertTrue(all(r.ok for r in res), [e.detail for r in res for e in r.failed])
            src = (root / _META).read_text(encoding="utf-8")
            # the definitions, not the forward declarations near the top of the file
            arena = src[src.rindex("static ggml_backend_buffer_t ggml_backend_meta_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {"):
                        src.rindex("static ggml_backend_buffer_t ggml_backend_meta_buffer_type_alloc_buffer_n(ggml_backend_buffer_type_t buft, ggml_tensor ** tensors, int n_tensors) {")]
            self.assertIn("BIGCHERRY_META_MEM compute dev=", arena)
            self.assertLess(arena.index("bufs.push_back(ggml_backend_buft_alloc_buffer("), arena.index("BIGCHERRY_META_MEM compute dev="))
            self.assertEqual(src.count("BIGCHERRY_META_MEM static dev="), 1)
            self.assertNotIn("BIGCHERRY_META_MEM static dev=", arena)
            self.assertIn("#include <cstdlib> // BigCherry 1339: getenv", src)
            # fusion overlap counters: defined before the check, one count per call and one per refusal
            cuda = (root / _CUDA).read_text(encoding="utf-8")
            check = cuda[cuda.index("static bool ggml_cuda_check_fusion_memory_ranges("):]
            self.assertLess(cuda.index("struct bc_fusion_overlap_stats_t {"), cuda.index("static bool ggml_cuda_check_fusion_memory_ranges("))
            self.assertLess(check.index("bc_fusion_overlap_stats.checks++;"), check.index("bc_fusion_overlap_stats.refused++;"))
            self.assertEqual(cuda.count("bc_fusion_overlap_stats.refused++;"), 1)
            before = {p: (root / p).read_text(encoding="utf-8") for p in _SRC}
            again = apply_all(_PATCHES, root)
            self.assertTrue(all(r.ok for r in again))
            self.assertEqual(before, {p: (root / p).read_text(encoding="utf-8") for p in _SRC})

    def test_changed_allocation_loop_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._root(td, {_META: _SRC[_META].replace("max_size = std::max(max_size, ggml_backend_buffer_get_size(bufs.back()));", "max_size = 0;")})
            res = apply_all(_PATCHES, root)
            self.assertFalse(all(r.ok for r in res))


if __name__ == "__main__":
    unittest.main()
