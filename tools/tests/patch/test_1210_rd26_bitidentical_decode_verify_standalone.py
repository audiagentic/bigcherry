"""Mechanics tests for 1210_rd26_bitidentical_decode_verify_standalone.

Only the ggml-cuda.cu edits are covered here (both MMVF-decision call sites:
the plain dispatch decision and the op-fusion decision gate added 2026-09-28
per PRBE20's found gap). The other three FilePatch entries (sgemm.cpp,
fattn.cu, fattn-tile.cuh, mmvq.cu) are unaffected by this change and are not
re-tested here.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bigcherry.patcher import apply_all  # noqa: E402

_REPO = Path(__file__).resolve().parents[3]
_PATCH_FILE = _REPO / "patches/1210_rd26_bitidentical_decode_verify_standalone/patch.py"
_spec = importlib.util.spec_from_file_location("patch_1210", _PATCH_FILE)
assert _spec is not None and _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

_GGML_CUDA_CU_SOURCE = r'''
static void ggml_cuda_op_mul_mat_vec_f(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    if (ggml_cuda_should_use_mmvf(src0->type, cc, warp_size, src0->ne, src0->nb, ne11)) {
        launch_mul_mat_vec_f(ctx, src0, src1, dst);
        return;
    }
    launch_mul_mat_f(ctx, src0, src1, dst);
}

static bool ggml_cuda_should_fuse_mul_mat_vec_f(const ggml_tensor * tensor) {
    ggml_tensor *       src0 = tensor->src[0];
    ggml_tensor *       src1 = tensor->src[1];
    const ggml_tensor * dst  = tensor;

    const bool is_mul_mat_id = tensor->op == GGML_OP_MUL_MAT_ID;

    bool use_mul_mat_vec_f =
        (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16) &&
        src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32;

    const int cc        = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    use_mul_mat_vec_f = use_mul_mat_vec_f && ggml_cuda_should_use_mmvf(src0->type, cc, warp_size, src0->ne, src0->nb, is_mul_mat_id ? src1->ne[2] : src1->ne[1]);

    //we only support fusion for ncols_dst = 1
    if (tensor->op == GGML_OP_MUL_MAT && dst->ne[1] != 1) {
        return false;
    }

    return use_mul_mat_vec_f;
}
'''


class Patch1210GgmlCudaCuMechanics(unittest.TestCase):
    def _tree(self, source: str = _GGML_CUDA_CU_SOURCE):
        td = tempfile.TemporaryDirectory()
        root = Path(td.name)
        path = root / "ggml/src/ggml-cuda/ggml-cuda.cu"
        path.parent.mkdir(parents=True)
        path.write_text(source, encoding="utf-8")
        return td, root, path

    def _ggml_cuda_cu_filepatch(self):
        for fp in _module.PATCHES:
            if fp.path == "ggml/src/ggml-cuda/ggml-cuda.cu":
                return fp
        raise AssertionError("ggml-cuda.cu FilePatch not found in 1210's PATCHES")

    def test_apply_and_idempotent(self):
        td, root, path = self._tree()
        with td:
            fp = self._ggml_cuda_cu_filepatch()
            first = apply_all([fp], root)
            self.assertTrue(all(r.ok for r in first), [e.detail for r in first for e in r.failed])
            text = path.read_text()
            # Plain dispatch call site normalized (pre-existing edit).
            self.assertIn("const int64_t ne11_mmvf = ne11 <= MMVF_MAX_BATCH_SIZE ? 1 : ne11;", text)
            # Fusion-decision call site normalized (new edit).
            self.assertIn("const int64_t ne11_fuse = is_mul_mat_id ? src1->ne[2] : src1->ne[1];", text)
            self.assertIn(
                "const int64_t ne11_fuse_mmvf = ne11_fuse <= MMVF_MAX_BATCH_SIZE ? 1 : ne11_fuse;",
                text,
            )
            self.assertIn(
                "ggml_cuda_should_use_mmvf(src0->type, cc, warp_size, src0->ne, src0->nb, ne11_fuse_mmvf)",
                text,
            )
            before = text
            second = apply_all([fp], root)
            self.assertTrue(all(r.ok for r in second), [e.detail for r in second for e in r.failed])
            self.assertEqual(before, path.read_text())

    def test_fusion_gate_edit_fails_closed_when_anchor_missing(self):
        # A source shape where the fusion-decision function does not exist at
        # all (e.g. an older/different upstream shape) must fail closed for
        # THAT edit specifically, not silently no-op or corrupt the file.
        source_without_fusion_gate = _GGML_CUDA_CU_SOURCE.replace(
            "static bool ggml_cuda_should_fuse_mul_mat_vec_f",
            "static bool renamed_should_fuse_mul_mat_vec_f",
        ).replace(
            "use_mul_mat_vec_f = use_mul_mat_vec_f && ggml_cuda_should_use_mmvf(src0->type, cc, warp_size, src0->ne, src0->nb, is_mul_mat_id ? src1->ne[2] : src1->ne[1]);",
            "use_mul_mat_vec_f = use_mul_mat_vec_f && something_else_entirely(src0, src1);",
        )
        td, root, path = self._tree(source_without_fusion_gate)
        with td:
            fp = self._ggml_cuda_cu_filepatch()
            results = apply_all([fp], root)
            self.assertFalse(all(r.ok for r in results))
            failed_ids = {e.edit_id for r in results for e in r.failed}
            self.assertIn("rd26a-mmvf-fusion-decode-verify", failed_ids)
            # The sibling plain-dispatch edit is unaffected by this shape change.
            self.assertNotIn("rd26a-mmvf-decode-verify", failed_ids)


if __name__ == "__main__":
    unittest.main()
