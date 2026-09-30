"""Offline mechanics tests for 1274_mmvq_kquant_f32_decode."""

from __future__ import annotations

import importlib.util
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bigcherry.patcher import apply_all  # noqa: E402

_REPO = Path(__file__).resolve().parents[3]
_PATCH_FILE = _REPO / "patches/1274_mmvq_kquant_f32_decode/patch.py"
_BASE_PATCH_FILE = _REPO / "patches/1241_rd33_mmvq_q8_0_f32_decode/patch.py"
_VENDOR = _REPO / "tools/lab/iq-mmvq/vendor-b11233"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_module = _load("patch_1274", _PATCH_FILE)
_base = _load("patch_1241_for_1274", _BASE_PATCH_FILE)
# 1241 REQUIRES 0600 (MMVQ geometry adds nwarps_explicit/rows_per_block_explicit), so the
# test tree carries 0600 first, exactly like any real build stack.
_geometry = _load("patch_0600_for_1274", _REPO / "patches/0600_mmvq_geometry/patch.py")


class Patch1274Mechanics(unittest.TestCase):
    def _tree(self):
        td = tempfile.TemporaryDirectory()
        root = Path(td.name)
        cuda = root / "ggml/src/ggml-cuda"
        cuda.mkdir(parents=True)
        shutil.copy2(_VENDOR / "mmvq.cu", cuda / "mmvq.cu")
        shutil.copy2(_VENDOR / "vecdotq.cuh", cuda / "vecdotq.cuh")
        geometry = apply_all([_geometry.PATCH], root)
        assert all(r.ok for r in geometry), [e.detail for r in geometry for e in r.failed]
        return td, root, cuda / "mmvq.cu", cuda / "vecdotq.cuh"

    def test_composes_after_1241_and_is_idempotent(self):
        td, root, mmvq, vecdot = self._tree()
        with td:
            vecdot_pristine = vecdot.read_text(encoding="utf-8")
            base = apply_all(_base.PATCHES, root)
            self.assertTrue(all(r.ok for r in base), [e.detail for r in base for e in r.failed])
            results = apply_all(_module.PATCHES, root)
            self.assertTrue(all(r.ok for r in results), [e.detail for r in results for e in r.failed])
            text = mmvq.read_text(encoding="utf-8")

            self.assertIn("vec_dot_f32_decode<type>", text)
            self.assertIn("static __device__ __forceinline__ float vec_dot_q4_K_f32(", text)
            self.assertIn("static __device__ __forceinline__ float vec_dot_q6_K_f32(", text)
            self.assertIn("ggml_cuda_mmvq_f32_decode<GGML_TYPE_Q8_0", text)
            self.assertIn("ggml_cuda_mmvq_f32_decode<ktype, 1>", text)
            self.assertIn("type=q4_k ncols=1", text)
            self.assertIn("type=q6_k ncols=1", text)
            self.assertIn("src0->type == GGML_TYPE_Q4_K || src0->type == GGML_TYPE_Q6_K", text)
            self.assertIn("ggml_cuda_pool_alloc<char> src1_q8_1", text)
            self.assertIn("BIGCHERRY_PATCH_HIT patch=1241_rd33 path=q8_0_f32_decode", text)
            self.assertEqual(vecdot_pristine, vecdot.read_text(encoding="utf-8"))

            before = text
            second = apply_all(_module.PATCHES, root)
            self.assertTrue(all(r.ok for r in second), [e.detail for r in second for e in r.failed])
            self.assertEqual(before, mmvq.read_text(encoding="utf-8"))

    def test_requires_1241_shape(self):
        td, root, mmvq, _ = self._tree()
        with td:
            pristine = mmvq.read_text(encoding="utf-8")
            results = apply_all(_module.PATCHES, root)
            self.assertFalse(all(r.ok for r in results))
            self.assertEqual(pristine, mmvq.read_text(encoding="utf-8"))

    def test_native_lane_mapping_is_preserved(self):
        td, root, mmvq, _ = self._tree()
        with td:
            self.assertTrue(all(r.ok for r in apply_all(_base.PATCHES, root)))
            self.assertTrue(all(r.ok for r in apply_all(_module.PATCHES, root)))
            text = mmvq.read_text(encoding="utf-8")
            self.assertIn("const int bq8_offset = QR4_K * ((iqs/2) / (QI8_1/2));", text)
            self.assertIn("bq4_K->qs + 16*bq8_offset + 4*((iqs/2)%4)", text)
            self.assertIn("const int bq8_offset = 2*QR6_K*(iqs/(QI6_K/2))", text)
            self.assertIn("const int vi = __vsubss4((vil | vih), 0x20202020);", text)

    def test_edit_contracts_are_fail_closed(self):
        self.assertEqual(1, len(_module.PATCHES))
        file_patch = _module.PATCHES[0]
        self.assertEqual("none", file_patch.language)
        for edit in file_patch.edits:
            self.assertEqual(1, edit.expect_matches, edit.id)
            self.assertTrue(edit.guard, edit.id)
            self.assertTrue(edit.rationale, edit.id)

    def test_anchor_mutation_fails_closed(self):
        td, root, mmvq, _ = self._tree()
        with td:
            self.assertTrue(all(r.ok for r in apply_all(_base.PATCHES, root)))
            text = mmvq.read_text(encoding="utf-8")
            broken = text.replace(
                "bool f32_act = false>",
                "bool f32_act_mutated = false>",
                1,
            )
            self.assertNotEqual(text, broken)
            mmvq.write_text(broken, encoding="utf-8")
            results = apply_all(_module.PATCHES, root)
            self.assertFalse(all(r.ok for r in results))
            failures = [e for r in results for e in r.failed]
            self.assertTrue(any(e.edit_id == "kquant-f32-helpers" for e in failures))
            self.assertEqual(broken, mmvq.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
