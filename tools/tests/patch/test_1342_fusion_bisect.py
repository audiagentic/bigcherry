"""Offline mechanics tests for 1342_fusion_bisect (fusion bisect by first-node op, per-family counters)."""

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
_PIN = "HEAD"
_CUDA = "ggml/src/ggml-cuda/ggml-cuda.cu"


def _load():
    spec = importlib.util.spec_from_file_location("patch_1342", _REPO / "patches/1342_fusion_bisect/patch.py")
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
_SRC = _pinned(_CUDA)


@unittest.skipUnless(_SRC is not None, "pinned vendor repository not present")
class Patch1342Mechanics(unittest.TestCase):
    def _root(self, td, text):
        root = Path(td)
        (root / _CUDA).parent.mkdir(parents=True, exist_ok=True)
        (root / _CUDA).write_text(text, encoding="utf-8", newline="\n")
        return root

    def test_apply_and_idempotent(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._root(td, _SRC)
            res = apply_all(_P.PATCHES, root)
            self.assertTrue(all(r.ok for r in res), [e.detail for r in res for e in r.failed])
            src = (root / _CUDA).read_text(encoding="utf-8")
            fn = src[src.index("static int ggml_cuda_try_fuse("):]
            # the bisect check sits after the global switch and before any fusion is tried
            self.assertLess(fn.index("if (disable_fusion) {"), fn.index("BIGCHERRY_FUSION_SKIP_OPS"))
            self.assertLess(fn.index("BIGCHERRY_FUSION_SKIP_OPS"), fn.index("ggml_tensor * node = cgraph->nodes[i];"))
            self.assertLess(src.index("struct bc_fusion_taken_stats_t {"), src.index("static int ggml_cuda_try_fuse("))
            call = src[src.index("int nodes_to_skip = ggml_cuda_try_fuse(cuda_ctx, cgraph, i);"):]
            self.assertLess(call.index("if (nodes_to_skip != 0) {"), call.index("bc_fusion_taken_stats.by_op["))
            self.assertEqual(src.count("bc_fusion_taken_stats.by_op["), 1)
            again = apply_all(_P.PATCHES, root)
            self.assertTrue(all(r.ok for r in again))
            self.assertEqual(src, (root / _CUDA).read_text(encoding="utf-8"))

    def test_changed_entry_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._root(td, _SRC.replace("    if (disable_fusion) {\n        return 0;\n    }\n", "    if (disable_fusion) return 0;\n", 1))
            res = apply_all(_P.PATCHES, root)
            self.assertFalse(all(r.ok for r in res))


if __name__ == "__main__":
    unittest.main()
