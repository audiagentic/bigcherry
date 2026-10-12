"""Offline mechanics tests for 1357_moe_router_splitk."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bigcherry.core import paths  # noqa: E402
from bigcherry.patcher import apply_all  # noqa: E402
from bigcherry.patch.pinned_source import copy_pinned  # noqa: E402

_REPO = Path(__file__).resolve().parents[3]
_V = paths.llama_root()
_F = "ggml/src/ggml-cuda/ggml-cuda.cu"
_G = "src/llama-graph.cpp"
_BC = "ggml/src/ggml-cuda/bc-moe-router-splitk.cuh"


def _load(name="1357_moe_router_splitk"):
    spec = importlib.util.spec_from_file_location("patch_" + name[:4], _REPO / "engines/llamacpp/patches" / name / "patch.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_P = _load()


@unittest.skipUnless((_V / _F).exists(), "pinned vendor checkout not present")
class Patch1357Mechanics(unittest.TestCase):
    def _root(self, td):
        root = Path(td)
        for rel in (_F, _G):
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            copy_pinned(_V / rel, root / rel)
        return root

    def test_apply_marks_router_and_dispatches_before_mmf(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._root(td)
            res = apply_all(_P.PATCHES, root)
            self.assertTrue(all(r.ok for r in res), [e.detail for r in res for e in r.failed])
            src = (root / _F).read_text(encoding="utf-8")
            graph = (root / _G).read_text(encoding="utf-8")
            owned = (root / _BC).read_text(encoding="utf-8")
            self.assertEqual(owned, _P._N_HELPERS)
            self.assertEqual(src.count('#include "bc-moe-router-splitk.cuh"'), 1)
            self.assertNotIn("bc_moe_router_splitk_partial(", src)

            # the marker follows the logits construction, only on the bare matmul over gate_inp, and uses slot 6
            mark = graph.index("if (logits->op == GGML_OP_MUL_MAT && logits->src[0] == gate_inp) {")
            self.assertLess(graph.index("logits = build_lora_mm(gate_inp, cur);"), mark)
            self.assertLess(mark, graph.index('cb(logits, "ffn_moe_logits", il);'))
            self.assertEqual(graph.count("((int32_t *) logits->op_params)[6] = 0x42435253;"), 1)
            self.assertIn("((const int32_t *) dst->op_params)[6] == 0x42435253", owned)

            fn = src[src.index("static void ggml_cuda_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {"):]
            fn = fn[:fn.index("\n}\n")]
            hit = "if (bc_moe_router_splitk() && bc_moe_router_marked(dst) && GGML_CUDA_CC_IS_AMD(cc)"
            self.assertEqual(fn.count(hit), 1)
            self.assertLess(fn.index(hit), fn.index("if (ggml_cuda_should_use_mmf(src0->type, cc, warp_size, src0->ne, src0->nb, ne11, /*mul_mat_id =*/ false)) {"))
            self.assertIn("src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32", fn)
            self.assertIn("ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(dst)", fn)

            # off unless the flag is set; the reduction adds the parts in ascending order, no atomics
            self.assertIn("return s != nullptr && atoi(s) != 0;", owned)
            self.assertIn("for (int z = 0; z < parts; ++z) {\n        sum += partial[((int64_t) z*N + n)*M + m];", owned)
            self.assertNotIn("atomic", owned)

            second = apply_all(_P.PATCHES, root)
            self.assertTrue(all(r.ok for r in second), [e.detail for r in second for e in r.failed])
            self.assertEqual(src, (root / _F).read_text(encoding="utf-8"))
            self.assertEqual(graph, (root / _G).read_text(encoding="utf-8"))
            self.assertEqual(owned, (root / _BC).read_text(encoding="utf-8"))

    def test_composes_after_1347(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._root(td)
            first = apply_all(_load("1347_f32_thin_transposed_mmvf").PATCHES, root)
            self.assertTrue(all(r.ok for r in first), [e.detail for r in first for e in r.failed])
            res = apply_all(_P.PATCHES, root)
            self.assertTrue(all(r.ok for r in res), [e.detail for r in res for e in r.failed])
            fn = (root / _F).read_text(encoding="utf-8")
            self.assertLess(fn.index("BIGCHERRY_PATCH_HIT patch=1347_f32_thin_transposed_mmvf"),
                            fn.index("BIGCHERRY_PATCH_HIT patch=1357_moe_router_splitk"))

    def test_changed_router_construction_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._root(td)
            p = root / _G
            before = p.read_text(encoding="utf-8").replace("logits = build_lora_mm(gate_inp, cur); // [n_expert, n_tokens]",
                                                           "logits = build_lora_mm(gate_inp, cur);")
            p.write_text(before, encoding="utf-8")
            res = apply_all(_P.PATCHES, root)
            self.assertFalse(all(r.ok for r in res))
            self.assertEqual(before, p.read_text(encoding="utf-8"))
            self.assertFalse((root / _BC).exists())

    def test_env_doc(self):
        self.assertEqual([doc.name for doc in _P.ENV_DOCS], ["BIGCHERRY_MOE_ROUTER_SPLITK"])


if __name__ == "__main__":
    unittest.main()
