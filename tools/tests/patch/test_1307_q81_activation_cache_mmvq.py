"""Offline mechanics tests for 1307_q81_activation_cache_mmvq (pinned mmvq.cu after 0600 + 1241; ggml-cuda.cu)."""

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
_VENDOR = _REPO / "tools/lab/iq-mmvq/vendor-b11233"
_GGML_CUDA = _REPO / "vendor/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_module = _load("patch_1307", _REPO / "patches/1307_q81_activation_cache_mmvq/patch.py")
_PATCHES = [p for p in _module.PATCHES if p.path in {"ggml/src/ggml-cuda/mmvq.cu", "ggml/src/ggml-cuda/ggml-cuda.cu"}]
_PREREQS = [
    _load("patch_0600", _REPO / "patches/0600_mmvq_geometry/patch.py").PATCH,
    *_load("patch_1241", _REPO / "patches/1241_rd33_mmvq_q8_0_f32_decode/patch.py").PATCHES,
]


@unittest.skipUnless(_GGML_CUDA.exists(), "pinned vendor checkout not present")
class Patch1307Mechanics(unittest.TestCase):
    def _tree(self):
        td = tempfile.TemporaryDirectory()
        root = Path(td.name)
        cuda = root / "ggml/src/ggml-cuda"
        cuda.mkdir(parents=True)
        for name in ("mmvq.cu", "mmvq.cuh", "vecdotq.cuh"):
            copy_pinned(_VENDOR / name, cuda / name)
        copy_pinned(_GGML_CUDA, cuda / "ggml-cuda.cu")
        prereq = apply_all(_PREREQS, root)
        assert all(r.ok for r in prereq), [e.detail for r in prereq for e in r.failed]
        return td, root, cuda

    def test_apply_and_idempotent(self):
        td, root, cuda = self._tree()
        with td:
            results = apply_all(_PATCHES, root)
            self.assertTrue(all(r.ok for r in results), [e.detail for r in results for e in r.failed])
            mmvq = (cuda / "mmvq.cu").read_text(encoding="utf-8")
            cu = (cuda / "ggml-cuda.cu").read_text(encoding="utf-8")
            self.assertIn('#include "hip-q81-cache.h"', mmvq)
            self.assertIn("ggml_hip_q81_cache_find(q81, key)", mmvq)
            self.assertIn("ggml_hip_q81_cache_publish(q81, key, r);", mmvq)
            # flattened-reshape lookup (1312 GDN final_output) after the padding-free reshape lookup
            self.assertLess(mmvq.index("hit = ggml_hip_q81_cache_find(q81, vkey);"), mmvq.index("hit = ggml_hip_q81_cache_find(q81, fkey);"))
            self.assertIn("src0->data, src0->type, src1_q8_1_ptr, ids_d", mmvq)
            self.assertNotIn("src1_q8_1(ctx.pool(), ne13*ne12", mmvq)  # pool buffer only on fallback now
            # publish must follow the quantize that fills the reservation (1235's contract)
            q = mmvq.index("quantize_row_q8_1_cuda(src1_d, nullptr, r.ptr")
            self.assertLess(q, mmvq.index("ggml_hip_q81_cache_publish(q81, key, r);"))
            self.assertIn("ggml_hip_q81_cache_begin_generation(", cu)
            self.assertIn("ggml_hip_q81_cache_set_capture_active(ggml_hip_q81_cache_for_context(*cuda_ctx), true);", cu)
            self.assertIn("ggml_hip_q81_cache_set_capture_active(ggml_hip_q81_cache_for_context(*cuda_ctx), false);", cu)

            second = apply_all(_PATCHES, root)
            self.assertTrue(all(r.ok for r in second), [e.detail for r in second for e in r.failed])
            self.assertEqual(mmvq, (cuda / "mmvq.cu").read_text(encoding="utf-8"))
            self.assertEqual(cu, (cuda / "ggml-cuda.cu").read_text(encoding="utf-8"))


# PA44-E packaging equivalence is part of the 1307 mechanics gate. Importing
# the TestCase here makes unittest's changed-patch loader execute the materialized
# pre/post merge tree checks instead of leaving the standalone regression undiscovered.
from tools.tests.patch.test_pa44e_merged_family_identity import PA44EMergedFamilyIdentity  # noqa: E402,F401


if __name__ == "__main__":
    unittest.main()
