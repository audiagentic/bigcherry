"""Mechanics tests for 1267_rd07_q6k_mmq_scale_fold."""

from __future__ import annotations
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bigcherry.patcher import apply_all  # noqa: E402
_REPO = Path(__file__).resolve().parents[3]
_PATCH_FILE = _REPO / "patches/1267_rd07_q6k_mmq_scale_fold/patch.py"
_spec = importlib.util.spec_from_file_location("patch_1267_rd07", _PATCH_FILE)
assert _spec is not None and _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

_VEC = _module._DF_HOIST_OLD + "\n" + _module._SC_FOLD_OLD + "\n" + "                for (int l = 0; l < tile_C::ne; ++l) {\n                    const int i = i0 + n*tile_C::I + tile_C::get_i(l);\n                    const int8_t * sc = (const int8_t *) (x_sc + i*sram_stride + k00/16);\n                    sum[(j0/tile_C::J + n)*tile_C::ne + l] += C.x[l] * sc[k01/4] * x_df[i*sram_stride] * dB;\n                }\n"
_MMQ = "#include <cstdint>\n\nstatic void ggml_cuda_mul_mat_q_switch_type(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream) {\n    switch (args.type_x) {\n        case GGML_TYPE_Q6_K:\n            mul_mat_q_case<GGML_TYPE_Q6_K>(ctx, args, stream);\n            break;\n    }\n}\n"
_JMAX = "int f() {\n    int ret = std::min(ne11, int64_t(512));\n    ret -= ret % 8;\n}\n"
_PERF = "void f() {\n        test_cases.emplace_back(new test_l2_norm_batch(GGML_TYPE_F32, { n, 16, 16, 1 }, 4, 1e-12f, true));\n    }\n\n\n    return test_cases;\n}\n"

class Patch1267Mechanics(unittest.TestCase):
    def _tree(self):
        td = tempfile.TemporaryDirectory(); root = Path(td.name); base = root / "ggml/src/ggml-cuda"; base.mkdir(parents=True); (root / "tests").mkdir()
        (base / "mmq-vec-dot.cuh").write_text(_VEC); (base / "mmq.cu").write_text(_MMQ); (base / "mmq.cuh").write_text(_JMAX); (root / "tests/test-backend-ops.cpp").write_text(_PERF)
        return td, root
    def test_apply_and_idempotent(self):
        td, root = self._tree()
        with td:
            first = apply_all(_module.PATCHES, root); self.assertTrue(all(r.ok for r in first), [e.detail for r in first for e in r.failed])
            self.assertIn("x_s2_reg", (root / "ggml/src/ggml-cuda/mmq-vec-dot.cuh").read_text())
            self.assertIn("patch=1267_rd07_q6k_mmq_scale_fold", (root / "ggml/src/ggml-cuda/mmq.cu").read_text())
            before = {p: (root / p).read_text() for p in ("ggml/src/ggml-cuda/mmq-vec-dot.cuh", "ggml/src/ggml-cuda/mmq.cu", "ggml/src/ggml-cuda/mmq.cuh", "tests/test-backend-ops.cpp")}
            second = apply_all(_module.PATCHES, root); self.assertTrue(all(r.ok for r in second)); self.assertEqual(before, {p: (root / p).read_text() for p in before})
    def test_missing_anchor_fails_closed(self):
        td, root = self._tree()
        with td:
            p = root / "ggml/src/ggml-cuda/mmq.cu"; p.write_text(_MMQ.replace("GGML_TYPE_Q6_K", "GGML_TYPE_Q5_K"))
            results = apply_all(_module.PATCHES, root); self.assertFalse(all(r.ok for r in results))
    def test_ambiguous_anchor_fails_closed(self):
        td, root = self._tree()
        with td:
            p = root / "ggml/src/ggml-cuda/mmq.cuh"; p.write_text(_JMAX + _JMAX)
            results = apply_all(_module.PATCHES, root); self.assertFalse(all(r.ok for r in results)); detail = "\n".join(e.detail for r in results for e in r.failed); self.assertIn("matched 2 time(s)", detail)

if __name__ == "__main__": unittest.main()
