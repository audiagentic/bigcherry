"""Offline mechanics tests for 1344_dsv4_hc_grid_index."""

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
_F = "ggml/src/ggml-cuda/dsv4-hc.cu"
_BC = "ggml/src/ggml-cuda/bc-dsv4-hc-grid.cuh"


def _load(pid):
    spec = importlib.util.spec_from_file_location("patch_" + pid[:4], _REPO / "engines" / "llamacpp" / "patches" / pid / "patch.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_P = _load("1344_dsv4_hc_grid_index")
_P1307 = _load("1307_q81_activation_cache_mmvq")  # the merged Q8_1 family; carries the HC_PRE consumer


def _only(patches):
    return [p for p in patches if p.path == _F]


def _kernel(src, name):
    body = src[src.index("static __global__ void " + name + "("):]
    return body[:body.index("\n}\n") + 3]


@unittest.skipUnless((_V / _F).exists(), "pinned vendor checkout not present")
class Patch1344Mechanics(unittest.TestCase):
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

            self.assertEqual(src.count('#include "bc-dsv4-hc-grid.cuh"'), 1)
            self.assertNotIn("static __global__ void dsv4_hc_pre_grid_f32(", src)
            self.assertNotIn("static __global__ void dsv4_hc_post_grid_f32(", src)
            self.assertEqual(owned, _P._N_KERNELS)

            pre = _kernel(owned, "dsv4_hc_pre_grid_f32")
            post = _kernel(owned, "dsv4_hc_post_grid_f32")
            for body in (pre, post):
                self.assertNotIn("%", body)
                self.assertNotIn(" / ", body.replace("1.0f / (1.0f + expf(", ""))
            self.assertIn("const int64_t it = blockIdx.y;", pre)
            self.assertIn("const int64_t idst = blockIdx.y;", post)
            self.assertIn("const int64_t it   = blockIdx.z;", post)

            flat_pre = _kernel(src, "dsv4_hc_pre_f32")
            flat_post = _kernel(src, "dsv4_hc_post_f32")
            self.assertEqual(pre[pre.index("    float sum = 0.0f;"):], flat_pre[flat_pre.index("    float sum = 0.0f;"):])
            self.assertEqual(post[post.index("    float sum = x[i0*sx0"):], flat_post[flat_post.index("    float sum = x[i0*sx0"):])
            self.assertIn("const int64_t i0 = ir % n_embd;", flat_pre)
            self.assertIn("return s == nullptr || atoi(s) != 0;", owned)
            self.assertIn("const bool bc_grid = bc_hc_grid_index() && n_tokens <= 65535;", src)
            self.assertIn("const bool bc_grid = bc_hc_grid_index() && hc <= 65535 && n_tokens <= 65535;", src)
            self.assertEqual(src.count("ggml_cuda_kernel_launch(kernel, bc_launch_params,"), 2)
            self.assertLess(src.index('#include "bc-dsv4-hc-grid.cuh"'), src.index("void ggml_cuda_op_dsv4_hc_comb("))

            second = apply_all(_P.PATCHES, root)
            self.assertTrue(all(r.ok for r in second), [e.detail for r in second for e in r.failed])
            self.assertEqual(src, (root / _F).read_text(encoding="utf-8"))
            self.assertEqual(owned, (root / _BC).read_text(encoding="utf-8"))

    def test_composes_after_1311(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._root(td)
            for patches in (_only(_P1307.PATCHES), _P.PATCHES):
                res = apply_all(patches, root)
                self.assertTrue(all(r.ok for r in res), [e.detail for r in res for e in r.failed])
            src = (root / _F).read_text(encoding="utf-8")
            self.assertEqual(src.count("ggml_cuda_kernel_launch(kernel, bc_launch_params,"), 2)
            self.assertTrue((root / _BC).is_file())

    def test_changed_launch_fails_closed_without_creating_owned_file(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._root(td)
            p = root / _F
            before = p.read_text(encoding="utf-8").replace(
                "    auto kernel = comb ? dsv4_hc_post_f32<true> : dsv4_hc_post_f32<false>;\n",
                "    auto kernel = comb != nullptr ? dsv4_hc_post_f32<true> : dsv4_hc_post_f32<false>;\n",
            )
            p.write_text(before, encoding="utf-8")
            res = apply_all(_P.PATCHES, root)
            self.assertFalse(all(r.ok for r in res))
            self.assertEqual(before, p.read_text(encoding="utf-8"))
            self.assertFalse((root / _BC).exists())

    def test_env_doc(self):
        self.assertEqual([doc.name for doc in _P.ENV_DOCS], ["BIGCHERRY_HC_GRID_INDEX"])


if __name__ == "__main__":
    unittest.main()
