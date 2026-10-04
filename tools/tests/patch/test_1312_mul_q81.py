"""Offline mechanics tests for 1312_mul_q81 (pinned ggml/src/ggml-cuda/binbcast.cu)."""

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
_VENDOR = _REPO / "vendor/llama.cpp/ggml/src/ggml-cuda/binbcast.cu"
_spec = importlib.util.spec_from_file_location("patch_1312", _REPO / "patches/1312_mul_q81/patch.py")
assert _spec is not None and _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


@unittest.skipUnless(_VENDOR.exists(), "pinned vendor checkout not present")
class Patch1312Mechanics(unittest.TestCase):
    def test_apply_and_idempotent(self):
        td = tempfile.TemporaryDirectory()
        with td:
            root = Path(td.name)
            path = root / "ggml/src/ggml-cuda/binbcast.cu"
            path.parent.mkdir(parents=True)
            shutil.copy2(_VENDOR, path)
            results = apply_all(_module.PATCHES, root)
            self.assertTrue(all(r.ok for r in results), [e.detail for r in results for e in r.failed])
            out = path.read_text(encoding="utf-8")
            self.assertIn("static __global__ void bc_mul_q81_kernel(", out)
            self.assertIn("!ggml_are_same_shape(a, dst) || !ggml_are_same_shape(b, dst) || dst->ne[0] % QK8_1 != 0", out)
            self.assertIn("q81, dst, dst->data, ctx.curr_stream_no, ne0, ne0_padded, ne1, ne2, ne3, ne0, ne0*ne1, ne0*ne1*ne2", out)
            # the hook is the first statement of ggml_cuda_op_mul, before its bin_bcast launch
            op = out.index("void ggml_cuda_op_mul(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {")
            self.assertLess(out.index("static bool bc_mul_q81_try("), op)
            hook = out.index("bc_mul_q81_try(ctx, dst)", op)
            self.assertLess(hook, out.index("bin_bcast_cuda<op_mul>", op))
            self.assertEqual(out.count("bc_mul_q81_try(ctx, dst)"), 1)
            second = apply_all(_module.PATCHES, root)
            self.assertTrue(all(r.ok for r in second), [e.detail for r in second for e in r.failed])
            self.assertEqual(out, path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
