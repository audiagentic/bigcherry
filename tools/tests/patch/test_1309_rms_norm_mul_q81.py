"""Offline mechanics tests for 1309_rms_norm_mul_q81 (pinned ggml/src/ggml-cuda/norm.cu)."""

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
_VENDOR = _REPO / "vendor/llama.cpp/ggml/src/ggml-cuda/norm.cu"
_spec = importlib.util.spec_from_file_location("patch_1309", _REPO / "patches/1309_rms_norm_mul_q81/patch.py")
assert _spec is not None and _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


@unittest.skipUnless(_VENDOR.exists(), "pinned vendor checkout not present")
class Patch1309Mechanics(unittest.TestCase):
    def test_apply_and_idempotent(self):
        td = tempfile.TemporaryDirectory()
        with td:
            root = Path(td.name)
            path = root / "ggml/src/ggml-cuda/norm.cu"
            path.parent.mkdir(parents=True)
            copy_pinned(_VENDOR, path)
            results = apply_all(_module.PATCHES, root)
            self.assertTrue(all(r.ok for r in results), [e.detail for r in results for e in r.failed])
            out = path.read_text(encoding="utf-8")
            self.assertIn("static __global__ void rms_norm_mul_q81_f32(", out)
            self.assertIn("warp_reduce_max<QK8_1>(fabsf(v))", out)
            self.assertIn("b.ds = make_half2(d, sum);", out)
            self.assertIn("ggml_hip_q81_cache_publish(q81, key, r);", out)
            self.assertIn("#include <atomic>", out)
            # key must mirror mmvq's: src1 node == mul_tensor, padded ne0, contiguous strides
            self.assertIn("q81, mul_tensor, mul_tensor->data, ctx.curr_stream_no,", out)
            # the q81 kernel is defined before ggml_cuda_op_rms_norm_fused uses it, and the gate precedes the
            # original launch it replaces (early return)
            self.assertLess(out.index("static __global__ void rms_norm_mul_q81_f32("),
                            out.index("void ggml_cuda_op_rms_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor)"))
            gate = out.index("bigcherry 1309 (PRBE06): decode-shaped fused")
            self.assertLess(gate, out.index("    rms_norm_mul_f32_cuda(src0_d, mul_d, nullptr, dst_d,", gate))
            second = apply_all(_module.PATCHES, root)
            self.assertTrue(all(r.ok for r in second), [e.detail for r in second for e in r.failed])
            self.assertEqual(out, path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
