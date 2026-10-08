"""Offline mechanics tests for 1347_f32_thin_transposed_mmvf."""

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
_F = "ggml/src/ggml-cuda/ggml-cuda.cu"
_BC = "ggml/src/ggml-cuda/bc-f32-thin-transposed-mmvf.cuh"


def _load():
    spec = importlib.util.spec_from_file_location("patch_1347", _REPO / "patches/1347_f32_thin_transposed_mmvf/patch.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_P = _load()


@unittest.skipUnless((_V / _F).exists(), "pinned vendor checkout not present")
class Patch1347Mechanics(unittest.TestCase):
    def _root(self, td):
        root = Path(td)
        (root / _F).parent.mkdir(parents=True, exist_ok=True)
        copy_pinned(_V / _F, root / _F)
        return root

    def test_apply_owned_file_and_idempotent(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._root(td)
            res = apply_all(_P.PATCHES, root)
            self.assertTrue(all(r.ok for r in res), [e.detail for r in res for e in r.failed])
            src = (root / _F).read_text(encoding="utf-8")
            owned = (root / _BC).read_text(encoding="utf-8")
            self.assertEqual(src.count('#include "bc-f32-thin-transposed-mmvf.cuh"'), 1)
            self.assertNotIn("static __global__ void bc_f32_thin_transpose(", src)
            self.assertEqual(owned, _P._N_HELPERS)
            self.assertIn("dst[m + n*stride_col_dst] = src[n + m*n_cols];", owned)
            self.assertIn("return s == nullptr || atoi(s) != 0;", owned)

            fn = src[src.index("static void ggml_cuda_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {"):]
            fn = fn[:fn.index("\n}\n")]
            thin = "if (bc_f32_thin_mmvf() && ne01 >= 2 && ne01 <= MMVF_MAX_BATCH_SIZE && ne11 > MMVF_MAX_BATCH_SIZE && ne2 == 1 && ne3 == 1"
            self.assertLess(fn.index("if (ne01 == 1 && ne11 > MMVF_MAX_BATCH_SIZE && ne2 == 1 && ne3 == 1"), fn.index(thin))
            self.assertLess(fn.index(thin), fn.index("if (ggml_cuda_should_use_mmf(src0->type, cc, warp_size, src0->ne, src0->nb, ne11, /*mul_mat_id =*/ false)) {"))
            self.assertEqual(src.count(thin), 1)
            self.assertIn("ggml_cuda_mul_mat_vec_f(ctx, src1, src0, nullptr, &dst_t);", fn)
            self.assertIn("dst_t.ne[0] = ne11;", fn)
            self.assertIn("dst_t.ne[1] = ne01;", fn)
            self.assertIn("ggml_cuda_pool_alloc<float> bc_dst_t(ctx.pool(), ne11*ne01);", fn)
            self.assertIn("&& ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(dst)\n            && ggml_cuda_should_use_mmvf(src1->type, cc, warp_size, src1->ne, src1->nb, /*ne11 =*/ 1)) {\n        static bool bc_logged", fn)
            self.assertIn("    ggml_cuda_mul_mat_cublas(ctx, src0, src1, dst);", fn)

            second = apply_all(_P.PATCHES, root)
            self.assertTrue(all(r.ok for r in second), [e.detail for r in second for e in r.failed])
            self.assertEqual(src, (root / _F).read_text(encoding="utf-8"))
            self.assertEqual(owned, (root / _BC).read_text(encoding="utf-8"))

    def test_changed_dispatch_fails_closed_without_creating_owned_file(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._root(td)
            p = root / _F
            before = p.read_text(encoding="utf-8").replace(
                "src0->nb, ne11, /*mul_mat_id =*/ false)) {\n        ggml_cuda_mul_mat_f(ctx, src0, src1, nullptr, dst);",
                "src0->nb, ne11, false)) {\n        ggml_cuda_mul_mat_f(ctx, src0, src1, nullptr, dst);",
            )
            p.write_text(before, encoding="utf-8")
            res = apply_all(_P.PATCHES, root)
            self.assertFalse(all(r.ok for r in res))
            self.assertEqual(before, p.read_text(encoding="utf-8"))
            self.assertFalse((root / _BC).exists())

    def test_env_doc(self):
        self.assertEqual([doc.name for doc in _P.ENV_DOCS], ["BIGCHERRY_F32_THIN_MMVF"])


if __name__ == "__main__":
    unittest.main()
