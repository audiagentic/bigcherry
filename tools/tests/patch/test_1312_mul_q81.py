"""Offline mechanics tests for 1312_mul_q81 (pinned ggml/src/ggml-cuda/unary.cu, on top of 1310)."""

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
_VENDOR = _REPO / "vendor/llama.cpp/ggml/src/ggml-cuda/unary.cu"


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, _REPO / rel)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_P1310 = _load("patch_1310", "patches/1310_act_q81/patch.py")
_P1312 = _load("patch_1312", "patches/1312_mul_q81/patch.py")


@unittest.skipUnless(_VENDOR.exists(), "pinned vendor checkout not present")
class Patch1312Mechanics(unittest.TestCase):
    def _tree(self, td: str) -> Path:
        path = Path(td) / "ggml/src/ggml-cuda/unary.cu"
        path.parent.mkdir(parents=True)
        copy_pinned(_VENDOR, path)
        return path

    def test_apply_after_1310_and_idempotent(self):
        with tempfile.TemporaryDirectory() as td:
            path = self._tree(td)
            root = Path(td)
            self.assertTrue(all(r.ok for r in apply_all(_P1310.PATCHES, root)))
            results = apply_all(_P1312.PATCHES, root)
            self.assertTrue(all(r.ok for r in results), [e.detail for r in results for e in r.failed])
            out = path.read_text(encoding="utf-8")
            impl = out.index("static void ggml_cuda_op_unary_mul_impl(")
            hook = out.index("bc_act_q81_try<op, true>(ctx, mul_node,", impl)
            # helper (1310) precedes the hook; the hook precedes the unchanged F32 launch in the same function
            self.assertLess(out.index("static bool bc_act_q81_try("), hook)
            self.assertLess(hook, out.index("(float *) mul_node->data, k, nc,", hook))
            self.assertLess(hook, out.index("void ggml_cuda_op_unary_mul(ggml_backend_cuda_context & ctx", impl))
            self.assertEqual(out.count("bc_act_q81_try<op, true>(ctx, mul_node,"), 1)
            self.assertIn("const bool flatten01 = mul_node->ne[0] % MATRIX_ROW_PADDING != 0 && mul_node->ne[1] > 1;", out)
            self.assertIn("const int64_t j0 = srow * o0 + scol;", out)
            second = apply_all(_P1312.PATCHES, root)
            self.assertTrue(all(r.ok for r in second), [e.detail for r in second for e in r.failed])
            self.assertEqual(out, path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
