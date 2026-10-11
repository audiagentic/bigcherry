"""Offline mechanics tests for 1281_moe_mul_mat_id_range."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bigcherry.core import paths  # noqa: E402
from bigcherry.patch.pinned_source import production_source  # noqa: E402
from bigcherry.patcher import apply_all  # noqa: E402

_REPO = Path(__file__).resolve().parents[3]
_V = paths.llama_root()
_PID = "1281_moe_mul_mat_id_range"
_CPU = "ggml/src/ggml-cpu/ggml-cpu.c"
_MMQ = "ggml/src/ggml-cuda/mmq.cu"
_MMVQ = "ggml/src/ggml-cuda/mmvq.cu"
_MMVF = "ggml/src/ggml-cuda/mmvf.cu"
_MMID = "ggml/src/ggml-cuda/mmid.cu"
_QCU = "ggml/src/ggml-cuda/quantize.cu"
_CUDA = "ggml/src/ggml-cuda/ggml-cuda.cu"
_BC_MMID = "ggml/src/ggml-cuda/bc-moe-range-mmid.cu"
_BC_Q = "ggml/src/ggml-cuda/bc-moe-range-quantize.cuh"
_BC_CUDA = "ggml/src/ggml-cuda/bc-moe-range-cuda.cuh"
_TEST = "tests/test-mul-mat-id-range.cpp"
_CREATED = {_BC_MMID, _BC_Q, _BC_CUDA, _TEST}


def _load():
    spec = importlib.util.spec_from_file_location("patch_1281", _REPO / "engines/llamacpp/patches" / _PID / "patch.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_P = _load()
_INPUTS = [fp.path for fp in _P.PATCHES if fp.path not in _CREATED] + [_MMID]
_ALL = _INPUTS + sorted(_CREATED)


@unittest.skipUnless((_V / "ggml/include/ggml.h").exists(), "pinned vendor repository not present")
class Patch1281Mechanics(unittest.TestCase):
    def _root(self, td, overrides=None):
        root = production_source(_INPUTS, Path(td), stop_before=_PID)
        for path, source in (overrides or {}).items():
            (root / path).write_text(source, encoding="utf-8", newline="\n")
        return root

    @staticmethod
    def _read(root, path):
        return (root / path).read_text(encoding="utf-8")

    def test_apply_composed_source_owned_files_and_idempotent(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._root(td)
            res = apply_all(_P.PATCHES, root)
            self.assertTrue(all(r.ok for r in res), [e.detail for r in res for e in r.failed])
            read = lambda p: self._read(root, p)  # noqa: E731

            self.assertEqual(read(_BC_MMID), _P._BC_MMID_CU)
            self.assertEqual(read(_BC_Q), _P._BC_Q8_RANGE_WRAPPER)
            self.assertEqual(read(_BC_CUDA), _P._BC_CUDA_POLICY)
            self.assertIn("static __global__ void mm_ids_range_translate(", read(_BC_MMID))
            self.assertIn("void ggml_cuda_mm_ids_range_translate(", read(_BC_MMID))
            self.assertIn("void quantize_scatter_range_mmq_q8_1_cuda(", read(_BC_Q))
            self.assertIn("static bool bc_cuda_mul_mat_id_range_supported(", read(_BC_CUDA))

            self.assertNotIn("static __global__ void mm_ids_range_translate(", read(_MMID))
            self.assertNotIn("void ggml_cuda_mm_ids_range_translate(", read(_MMID))
            self.assertIn("void ggml_cuda_mm_ids_range_translate(", read("ggml/src/ggml-cuda/mmid.cuh"))
            self.assertEqual(read(_QCU).count('#include "bc-moe-range-quantize.cuh"'), 1)
            self.assertNotIn("void quantize_scatter_range_mmq_q8_1_cuda(", read(_QCU))
            self.assertIn("void quantize_scatter_range_mmq_q8_1_cuda(", read("ggml/src/ggml-cuda/quantize.cuh"))
            self.assertEqual(read(_CUDA).count('#include "bc-moe-range-cuda.cuh"'), 1)
            self.assertNotIn("static bool bc_cuda_mul_mat_id_range_supported(", read(_CUDA))
            self.assertIn("bc_cuda_mul_mat_id_range_supported(op,", read(_CUDA))

            common = read("ggml/src/ggml-cuda/common.cuh")
            self.assertEqual(common.count("    int32_t id_base = 0;\n    int64_t id_count = 0;\n"), 2)
            self.assertEqual(read(_MMVQ).count("const int64_t bc_local = (int64_t) ids["), 2)
            self.assertEqual(read(_MMVF).count("const int64_t bc_local = (int64_t) channel_x - (int64_t) fusion.id_base;"), 1)
            for host in (read(_MMVQ), read(_MMVF)):
                self.assertEqual(host.count("CUDA_CHECK(cudaMemsetAsync(dst->data, 0, ggml_nbytes(dst), ctx.stream()));"), 1)

            mmq = read(_MMQ)
            self.assertIn("ggml_cuda_mm_ids_range_translate(bc_ids, bc_ids_local.get()", mmq)
            self.assertIn("ggml_cuda_launch_mm_ids_helper(bc_ids, ids_src1.get(), ids_dst.get(), expert_bounds.get(),", mmq)
            self.assertIn("const bool bc_range_dedup = bc_range && !use_native_fp4 && ne11 == 1 && n_expert_used > 1;", mmq)
            self.assertIn("quantize_scatter_range_mmq_q8_1_cuda(src1_d, ids_src1.get(), src1_q8_1.get()", mmq)

            quantize = read(_QCU)
            self.assertIn("template <mmq_q8_1_ds_layout ds_layout, bool scatter, bool range_scatter = false>", quantize)
            self.assertEqual(quantize.count("if (i == -1) {"), 1)
            self.assertNotIn("BIGCHERRY_MOE_RANGE_DEDUP", mmq)

            h, c, cpu = read("ggml/include/ggml.h"), read("ggml/src/ggml.c"), read(_CPU)
            for decl in ("ggml_mul_mat_id_range(", "ggml_mul_mat_id_is_range(", "ggml_mul_mat_id_range_base("):
                self.assertIn(decl, h)
            self.assertIn("struct ggml_tensor * result = ggml_mul_mat_id(ctx, as, b, ids);", c)
            self.assertIn("ggml_set_op_params_i32(result, 7, id_base);", c)
            self.assertIn("const int64_t bc_local = (int64_t) i02 - (int64_t) bc_id_base;", cpu)
            self.assertLess(cpu.index("memset(dst->data, 0, ggml_nbytes(dst));"),
                            cpu.index("    ggml_barrier(params->threadpool);"))
            repack = read("ggml/src/ggml-cpu/repack.cpp")
            self.assertEqual(repack.count("const int64_t bc_local = (int64_t) i02 - (int64_t) bc_id_base;"), 1)
            self.assertIn("llama_build_and_test(test-mul-mat-id-range.cpp)", read("tests/CMakeLists.txt"))
            self.assertIn("ggml_mul_mat_id_range(ctx, w, x, ids_global, c.id_base)", read(_TEST))

            before = {p: read(p) for p in _ALL}
            again = apply_all(_P.PATCHES, root)
            self.assertTrue(all(r.ok for r in again), [e.detail for r in again for e in r.failed])
            self.assertEqual(before, {p: read(p) for p in _ALL})

    def test_missing_composed_anchor_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._root(td)
            cpu = self._read(root, _CPU)
            old = "                const int32_t i02 = *(const int32_t *) ((const char *) ids->data + iid1*ids->nb[1] + id*ids->nb[0]);\n"
            self.assertEqual(cpu.count(old), 1)
            (root / _CPU).write_text(cpu.replace(old, old.replace("const int32_t", "int32_t"), 1),
                                     encoding="utf-8", newline="\n")
            res = apply_all(_P.PATCHES, root)
            self.assertFalse(all(r.ok for r in res))

    def test_owned_files_are_only_created_by_1281(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._root(td)
            for path in _CREATED:
                self.assertFalse((root / path).exists())
            res = apply_all(_P.PATCHES, root)
            self.assertTrue(all(r.ok for r in res), [e.detail for r in res for e in r.failed])
            for path in _CREATED:
                self.assertTrue((root / path).is_file())


if __name__ == "__main__":
    unittest.main()
