"""Offline mechanics tests for 1345_moe_ids_multiwarp."""

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
_F = "ggml/src/ggml-cuda/mmid.cu"
_BC = "ggml/src/ggml-cuda/bc-moe-ids-multiwarp.cuh"


def _load(pid):
    spec = importlib.util.spec_from_file_location("patch_" + pid[:4], _REPO / "engines" / "llamacpp" / "patches" / pid / "patch.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_P = _load("1345_moe_ids_multiwarp")
_P1281 = _load("1281_moe_mul_mat_id_range")


def _only(patches):
    return [p for p in patches if p.path == _F]


@unittest.skipUnless((_V / _F).exists(), "pinned vendor checkout not present")
class Patch1345Mechanics(unittest.TestCase):
    def _root(self, td):
        root = Path(td)
        (root / _F).parent.mkdir(parents=True, exist_ok=True)
        copy_pinned(_V / _F, root / _F)
        return root

    def _check(self, src, owned):
        native = src[src.index("static __global__ void mm_ids_helper("):src.index('#include "bc-moe-ids-multiwarp.cuh"')]
        mw = owned[owned.index("static __global__ void bc_mm_ids_helper_mw("):owned.index("static bool bc_mm_ids_multiwarp() {")]
        for line in (
            "const int iex_used = expert_used == expert ? iex : -1;",
            "nex_prev += expert_used < expert;",
            "const int it_compact_add_self = warp_reduce_any<neu_padded>(iex_used != -1);",
            "it_compact += __shfl_sync(0xFFFFFFFF, it_compact_add_lower + it_compact_add_self, warp_size - 1, warp_size);",
        ):
            self.assertIn(line, native)
            self.assertIn(line, mw)
        self.assertIn("const int row      = nex_prev_all + row_base + itc;", mw)
        self.assertIn("ids_dst[row] = it*n_expert_used + iex_used;", mw)
        self.assertIn("ids_src1[it*n_expert_used + iex_used] = row;", mw)
        self.assertIn("ids_src1[row] = it*sis1 + iex_used % nchannels_y;", mw)
        self.assertIn("expert_bounds[gridDim.x] = nex_prev_all + row_total;", mw)
        self.assertLess(mw.index("counts[warp]           = it_compact;"), mw.index("__syncthreads();"))
        self.assertLess(mw.index("__syncthreads();"), mw.index("const mm_ids_helper_store store_it = store_w[itc];"))
        self.assertIn("mm_ids_helper_store * store_w = store + t_begin;", mw)

        self.assertEqual(src.count('#include "bc-moe-ids-multiwarp.cuh"'), 1)
        self.assertNotIn("static __global__ void bc_mm_ids_helper_mw(", src)
        self.assertEqual(owned, _P._N_KERNEL)
        self.assertIn("return s == nullptr || atoi(s) != 0;", owned)
        self.assertIn("if (bc_mm_ids_multiwarp() && n_tokens >= 128) {", src)
        self.assertEqual(
            src.count(
                "launch_mm_ids_helper<10>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used,"
            ),
            1,
        )
        self.assertEqual(src.count("bc_launch_mm_ids_helper_mw<10>("), 1)
        self.assertNotIn("bc_launch_mm_ids_helper_mw< 0>", src)

    def test_apply_owned_file_and_idempotent(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._root(td)
            res = apply_all(_P.PATCHES, root)
            self.assertTrue(all(r.ok for r in res), [e.detail for r in res for e in r.failed])
            src = (root / _F).read_text(encoding="utf-8")
            owned = (root / _BC).read_text(encoding="utf-8")
            self._check(src, owned)
            second = apply_all(_P.PATCHES, root)
            self.assertTrue(all(r.ok for r in second), [e.detail for r in second for e in r.failed])
            self.assertEqual(src, (root / _F).read_text(encoding="utf-8"))
            self.assertEqual(owned, (root / _BC).read_text(encoding="utf-8"))

    def test_composes_after_1281(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._root(td)
            for patches in (_only(_P1281.PATCHES), _P.PATCHES):
                res = apply_all(patches, root)
                self.assertTrue(all(r.ok for r in res), [e.detail for r in res for e in r.failed])
            src = (root / _F).read_text(encoding="utf-8")
            owned = (root / _BC).read_text(encoding="utf-8")
            self._check(src, owned)
            self.assertIn("static __global__ void mm_ids_range_translate(", src)

    def test_changed_switch_fails_closed_without_creating_owned_file(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._root(td)
            p = root / _F
            before = p.read_text(encoding="utf-8").replace(
                "        case  2:\n            launch_mm_ids_helper< 2>(",
                "        case 2:\n            launch_mm_ids_helper< 2>(",
            )
            p.write_text(before, encoding="utf-8")
            res = apply_all(_P.PATCHES, root)
            self.assertFalse(all(r.ok for r in res))
            self.assertEqual(before, p.read_text(encoding="utf-8"))
            self.assertFalse((root / _BC).exists())

    def test_env_doc(self):
        self.assertEqual([doc.name for doc in _P.ENV_DOCS], ["BIGCHERRY_MOE_IDS_MULTIWARP"])


if __name__ == "__main__":
    unittest.main()
