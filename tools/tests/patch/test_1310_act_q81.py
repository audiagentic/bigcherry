"""Offline mechanics tests for 1310_act_q81 (pinned ggml/src/ggml-cuda/unary.cu)."""

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
_spec = importlib.util.spec_from_file_location("patch_1310", _REPO / "engines/llamacpp/patches/1307_q81_activation_cache_mmvq/patch.py")
assert _spec is not None and _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
_PATCHES = [p for p in _module.PATCHES if p.description.startswith("1310:")]


@unittest.skipUnless(_VENDOR.exists(), "pinned vendor checkout not present")
class Patch1310Mechanics(unittest.TestCase):
    def test_apply_and_idempotent(self):
        td = tempfile.TemporaryDirectory()
        with td:
            root = Path(td.name)
            path = root / "ggml/src/ggml-cuda/unary.cu"
            path.parent.mkdir(parents=True)
            copy_pinned(_VENDOR, path)
            results = apply_all(_PATCHES, root)
            self.assertTrue(all(r.ok for r in results), [e.detail for r in results for e in r.failed])
            out = path.read_text(encoding="utf-8")
            self.assertIn("static __global__ void bc_act_q81_kernel(", out)
            self.assertIn("bc_act_q81_try<op, false>(ctx, dst, (const float *) src0_d, nullptr, 0, 0, false)", out)
            self.assertIn("bc_act_q81_try<op, true>(ctx, dst, src0_p, src1_p,", out)
            self.assertIn("q81, dst, dst->data, ctx.curr_stream_no, ne0, ne0_padded, ne1, ne2, ne3, ne0, ne0*ne1, ne0*ne1*ne2", out)
            self.assertIn('rows=%lld\\n"', out)
            # helper precedes both callers; the gated hook sits immediately before the original F32 gated launch
            helper = out.index("static bool bc_act_q81_try(")
            self.assertLess(helper, out.index("void ggml_cuda_op_unary(ggml_backend_cuda_context & ctx, ggml_tensor * dst)"))
            gated = out.index("bc_act_q81_try<op, true>")
            self.assertLess(gated, out.index("unary_gated_cuda<op>(src0_p, src1_p, (float *)dst_d", gated))
            second = apply_all(_PATCHES, root)
            self.assertTrue(all(r.ok for r in second), [e.detail for r in second for e in r.failed])
            self.assertEqual(out, path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
