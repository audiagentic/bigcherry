"""Offline mechanics tests for 1281_moe_mul_mat_id_range (MET02 phase A)."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bigcherry.core import paths  # noqa: E402
from bigcherry.patcher import apply_all  # noqa: E402

_REPO = Path(__file__).resolve().parents[3]
_V = paths.llama_root()
_PIN = "HEAD"  # the vendor checkout is at the pinned revision
_NEW = "tests/test-mul-mat-id-range.cpp"
_CPU = "ggml/src/ggml-cpu/ggml-cpu.c"
_QCU = "ggml/src/ggml-cuda/quantize.cu"
_QH = "ggml/src/ggml-cuda/quantize.cuh"


def _load():
    spec = importlib.util.spec_from_file_location("patch_1281", _REPO / "engines/llamacpp/patches/1281_moe_mul_mat_id_range/patch.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_P = _load()
_FILES = [fp.path for fp in _P.PATCHES if fp.path != _NEW]


def _pinned(path):
    # the pristine pinned file from the vendor repository, so a patched working tree does not matter
    try:
        res = subprocess.run(["git", "-C", str(_V), "show", f"{_PIN}:{path}"], capture_output=True, text=True,
                             encoding="utf-8", check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return res.stdout


_SRC = {path: _pinned(path) for path in _FILES}


@unittest.skipUnless(all(text is not None for text in _SRC.values()), "pinned vendor repository not present")
class Patch1281Mechanics(unittest.TestCase):
    def _root(self, td, overrides=None):
        root = Path(td)
        for path, text in _SRC.items():
            (root / path).parent.mkdir(parents=True, exist_ok=True)
            (root / path).write_text((overrides or {}).get(path, text), encoding="utf-8", newline="\n")
        return root

    def test_apply_and_idempotent(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._root(td)
            res = apply_all(_P.PATCHES, root)
            self.assertTrue(all(r.ok for r in res), [e.detail for r in res for e in r.failed])
            read = lambda p: (root / p).read_text(encoding="utf-8")  # noqa: E731
            h, c, cpu, cuda = read("ggml/include/ggml.h"), read("ggml/src/ggml.c"), read(_CPU), read("ggml/src/ggml-cuda/ggml-cuda.cu")
            # the ordinary constructor is untouched and the variant is built on it
            self.assertEqual(c.count("struct ggml_tensor * ggml_mul_mat_id(\n"), 1)
            self.assertIn("struct ggml_tensor * result = ggml_mul_mat_id(ctx, as, b, ids);\n\n    ggml_set_op_params_i32(result, 6,", c)
            self.assertIn("ggml_set_op_params_i32(result, 7, id_base);", c)
            for decl in ("ggml_mul_mat_id_range(", "ggml_mul_mat_id_is_range(", "ggml_mul_mat_id_range_base("):
                self.assertIn(decl, h)
            # CPU: widened subtraction, skip before any grouping, output cleared before the barrier
            group = cpu[cpu.index("        if (bc_range) {\n            // BigCherry 1281: a lane whose expert"):cpu.index("    // reset current_chunk")]
            self.assertIn("memset(dst->data, 0, ggml_nbytes(dst));", group)
            self.assertIn("const int64_t bc_local = (int64_t) i02 - (int64_t) bc_id_base;", group)
            self.assertLess(group.index("continue; // not held here"), group.index("MMID_MATRIX_ROW(i02, matrix_row_counts[i02])"))
            self.assertLess(cpu.index("memset(dst->data, 0, ggml_nbytes(dst));"), cpu.index("    ggml_barrier(params->threadpool);\n\n    for (int cur_a = 0; cur_a < n_as; ++cur_a) {"))
            # the ordinary op keeps its range assertion
            self.assertIn("                assert(i02 >= 0 && i02 < n_as);\n", group)
            # the repacked-weights CPU path groups rows itself and gets the same translation
            repack = read("ggml/src/ggml-cpu/repack.cpp")
            self.assertEqual(repack.count("const int64_t bc_local = (int64_t) i02 - (int64_t) bc_id_base;"), 1)
            self.assertLess(repack.index("memset(dst->data, 0, ggml_nbytes(dst));"), repack.index("const int64_t bc_local"))
            # phase A: the GPU backend refuses the variant
            refuse = cuda.index("if (ggml_mul_mat_id_is_range(op)) {")
            self.assertIn("if (!bc_cuda_mul_mat_id_range_supported(op, ggml_cuda_info().devices[dev_ctx->device].cc)) {", cuda[refuse:refuse + 420])
            # the policy mirrors the dispatch order (MMVQ, MMVF, MMQ) and is defined before both users
            policy = cuda.index("static bool bc_cuda_mul_mat_id_range_supported(")
            self.assertLess(policy, cuda.index("static bool ggml_cuda_mul_mat_id_needs_sync("))
            self.assertLess(policy, refuse)
            # both fusion argument structs (host and device) carry the range; a guard collision once dropped one
            common = read("ggml/src/ggml-cuda/common.cuh")
            self.assertEqual(common.count("    int32_t id_base = 0;\n    int64_t id_count = 0;\n"), 2)
            # range nodes are never fused; the kernels translate and skip, the hosts clear dst first
            self.assertIn("ggml_mul_mat_id_is_range(ffn_up) || ggml_mul_mat_id_is_range(ffn_gate)", cuda)
            mmvq, mmvf, mmq, mmid = (read("ggml/src/ggml-cuda/" + f) for f in ("mmvq.cu", "mmvf.cu", "mmq.cu", "mmid.cu"))
            self.assertEqual(mmvq.count("const int64_t bc_local = (int64_t) ids["), 2)
            self.assertEqual(mmvf.count("const int64_t bc_local = (int64_t) channel_x - (int64_t) fusion.id_base;"), 1)
            for host in (mmvq, mmvf):
                self.assertEqual(host.count("CUDA_CHECK(cudaMemsetAsync(dst->data, 0, ggml_nbytes(dst), ctx.stream()));"), 1)
            self.assertIn("ids_local[i] = local >= 0 && local < n_local ? (int32_t) local : INT_MAX;", mmid)
            # QFP30 chunk 2: only the range Q8_1 scatter variant knows the -1 inactive-slot sentinel.
            quantize, quantize_h = read(_QCU), read(_QH)
            self.assertIn("void quantize_scatter_range_mmq_q8_1_cuda(", quantize)
            self.assertIn("void quantize_scatter_range_mmq_q8_1_cuda(", quantize_h)
            self.assertEqual(quantize.count("if (i == -1) {"), 1)
            self.assertIn(
                "if constexpr (range_scatter) {\n"
                "                // BigCherry 1281 (QFP30): -1 is the only inactive range inverse-map sentinel.\n"
                "                // Every other value follows the ordinary indexing path so corrupt maps are not silently hidden.\n"
                "                if (i == -1) {\n",
                quantize,
            )
            self.assertEqual(quantize.count("quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_D4, true, true>"), 1)
            self.assertEqual(quantize.count("quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_DS4, true, true>"), 1)
            self.assertEqual(quantize.count("quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_D2S6, true, true>"), 1)
            # Ordinary wrapper text is an exact pinned-source anchor and remains byte-for-byte present after patching.
            self.assertEqual(_SRC[_QCU].count(_P._A_Q8_SCATTER_WRAPPER), 1)
            self.assertEqual(quantize.count(_P._A_Q8_SCATTER_WRAPPER), 1)
            self.assertIn("quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_D4, true><<<", _P._A_Q8_SCATTER_WRAPPER)
            self.assertIn("quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_DS4, true><<<", _P._A_Q8_SCATTER_WRAPPER)
            self.assertIn("quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_D2S6, true><<<", _P._A_Q8_SCATTER_WRAPPER)
            self.assertIn("ggml_cuda_launch_mm_ids_helper(bc_ids, ids_src1.get(), ids_dst.get(), expert_bounds.get(),", mmq)
            self.assertIn("const bool dedup_bcast = ne11 == 1 && n_expert_used > 1 && (!bc_range || bc_range_dedup);", mmq)
            # QFP30: range dedup (no flag) - sentinel fill, range-only scatter call, ordinary call kept
            self.assertIn("const bool bc_range_dedup = bc_range && !use_native_fp4 && ne11 == 1 && n_expert_used > 1;", mmq)
            self.assertNotIn("BIGCHERRY_MOE_RANGE_DEDUP", mmq)
            self.assertIn("cudaMemsetAsync(ids_src1.get(), bc_range_dedup ? 0xff : 0, ne_get_rows*sizeof(int32_t), stream)", mmq)
            self.assertEqual(mmq.count("quantize_scatter_range_mmq_q8_1_cuda(src1_d, ids_src1.get(), src1_q8_1.get(), src0->type, ne10,"), 1)
            self.assertEqual(mmq.count("quantize_scatter_mmq_q8_1_cuda(src1_d, ids_src1.get(), src1_q8_1.get(), src0->type, ne10,"), 1)
            self.assertLess(mmq.index("} else if (dedup_bcast && bc_range_dedup) {"), mmq.index("        } else if (dedup_bcast) {"))
            self.assertIn("llama_build_and_test(test-mul-mat-id-range.cpp)", read("tests/CMakeLists.txt"))
            self.assertIn("ggml_mul_mat_id_range(ctx, w, x, ids_global, c.id_base)", read(_NEW))
            before = {p: read(p) for p in _FILES + [_NEW]}
            again = apply_all(_P.PATCHES, root)
            self.assertTrue(all(r.ok for r in again), [e.detail for r in again for e in r.failed])
            self.assertEqual(before, {p: read(p) for p in _FILES + [_NEW]})

    def test_changed_grouping_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._root(td, {_CPU: _SRC[_CPU].replace("                assert(i02 >= 0 && i02 < n_as);\n",
                                                            "                assert(i02 < n_as);\n")})
            res = apply_all(_P.PATCHES, root)
            self.assertFalse(all(r.ok for r in res))


if __name__ == "__main__":
    unittest.main()
