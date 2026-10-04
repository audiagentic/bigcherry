"""Offline mechanics tests for 1313_scale_act_fuse (pinned unary.cu/unary.cuh/ggml-cuda.cu, on top of 1310)."""

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
_CUDA = _REPO / "vendor/llama.cpp/ggml/src/ggml-cuda"
_FILES = ("unary.cu", "unary.cuh", "ggml-cuda.cu")


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, _REPO / rel)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_P1310 = _load("patch_1310", "patches/1310_act_q81/patch.py")
_P1313 = _load("patch_1313", "patches/1313_scale_act_fuse/patch.py")


@unittest.skipUnless((_CUDA / "unary.cu").exists(), "pinned vendor checkout not present")
class Patch1313Mechanics(unittest.TestCase):
    def test_apply_after_1310_and_idempotent(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            dst = root / "ggml/src/ggml-cuda"
            dst.mkdir(parents=True)
            for f in _FILES:
                shutil.copy2(_CUDA / f, dst / f)
            self.assertTrue(all(r.ok for r in apply_all(_P1310.PATCHES, root)))
            results = apply_all(_P1313.PATCHES, root)
            self.assertTrue(all(r.ok for r in results), [e.detail for r in results for e in r.failed])
            unary = (dst / "unary.cu").read_text(encoding="utf-8")
            # kernel sits after 1310's cache include and before the relu+sqr fusion
            self.assertLess(unary.index('#include "hip-q81-cache.h"'), unary.index("static __global__ void bc_scale_act_kernel("))
            self.assertLess(unary.index("void bc_scale_act_fused("), unary.index("/* fused relu + sqr */"))
            self.assertIn("const float scaled = s0 * x[i] + b0;", unary)
            header = (dst / "unary.cuh").read_text(encoding="utf-8")
            self.assertIn("void bc_scale_act_fused(ggml_backend_cuda_context & ctx, const ggml_tensor * scale0", header)
            cu = (dst / "ggml-cuda.cu").read_text(encoding="utf-8")
            softcap = cu.index("ggml_cuda_op_softcap(*cuda_ctx, cgraph->nodes[i + 2], node);")
            match = cu.index("bigcherry 1313: SCALE -> UNARY")
            self.assertLess(softcap, match)
            self.assertLess(match, cu.index("static void ggml_cuda_graph_evaluate_and_capture(", match))
            self.assertIn("!act_then_mul && ggml_can_fuse(cgraph, i, ops3, 2)", cu)
            before = {f: (dst / f).read_text(encoding="utf-8") for f in _FILES}
            second = apply_all(_P1313.PATCHES, root)
            self.assertTrue(all(r.ok for r in second), [e.detail for r in second for e in r.failed])
            for f in _FILES:
                self.assertEqual(before[f], (dst / f).read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
