"""Offline mechanics tests for 1301_prbe115_q8_f32_mtp_widths."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bigcherry.patcher import apply_all  # noqa: E402
from bigcherry.patch.pinned_source import copy_pinned  # noqa: E402
from bigcherry.core import paths  # noqa: E402

_REPO = Path(__file__).resolve().parents[3]
_LLAMA = paths.llama_root()
_VENDOR = _LLAMA / "ggml/src/ggml-cuda"  # pinned source (copy_pinned reads the HEAD commit)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_module = _load("patch_1301", _REPO / "patches/1301_prbe115_q8_f32_mtp_widths/patch.py")
# 1301 requires 1241 (which requires 0600); apply both to the fixture first.
_PREREQS = [
    _load("patch_0600", _REPO / "patches/0600_mmvq_geometry/patch.py").PATCH,
    *_load("patch_1241", _REPO / "patches/1241_rd33_mmvq_q8_0_f32_decode/patch.py").PATCHES,
]


class Patch1301Mechanics(unittest.TestCase):
    def _tree(self):
        td = tempfile.TemporaryDirectory()
        root = Path(td.name)
        cuda = root / "ggml/src/ggml-cuda"
        cuda.mkdir(parents=True)
        for name in ("mmvq.cu", "mmvq.cuh", "vecdotq.cuh"):
            copy_pinned(_VENDOR / name, cuda / name)
        prereq = apply_all(_PREREQS, root)
        assert all(r.ok for r in prereq), [e.detail for r in prereq for e in r.failed]
        return td, root, cuda / "mmvq.cu"

    def test_apply_widens_gate_and_dispatches_each_width(self):
        td, root, mmvq_path = self._tree()
        with td:
            results = apply_all(_module.PATCHES, root)
            self.assertTrue(all(r.ok for r in results), [e.detail for r in results for e in r.failed])
            mmvq = mmvq_path.read_text(encoding="utf-8")
            self.assertIn('getenv("BIGCHERRY_Q8_F32_MAXCOLS")', mmvq)
            self.assertIn("ne1 >= 1 && ne1 <= prbe115_maxcols", mmvq)
            self.assertIn("(prbe115_rdna4 && GGML_CUDA_CC_IS_RDNA4(rd33_cc))", mmvq)
            self.assertNotIn("&& ne1 == 1 && !forced.requested()) {\n        const int rd33_cc", mmvq)
            for w in range(1, 9):
                self.assertIn(f"case {w}: rd33_launch(std::integral_constant<int, {w}>{{}}); break;", mmvq)
            self.assertNotIn("            rd33_launch(std::integral_constant<int, 1>{});\n            return;", mmvq)

            before = mmvq
            second = apply_all(_module.PATCHES, root)
            self.assertTrue(all(r.ok for r in second), [e.detail for r in second for e in r.failed])
            self.assertEqual(before, mmvq_path.read_text(encoding="utf-8"))

    def test_missing_1241_gate_fails_closed_without_writes(self):
        td, root, mmvq_path = self._tree()
        with td:
            broken = mmvq_path.read_text(encoding="utf-8").replace(
                "src0->type == GGML_TYPE_Q8_0 && ne1 == 1 && !forced.requested()",
                "src0->type == GGML_TYPE_Q8_0 && ne1 == 2 && !forced.requested()",
                1,
            )
            mmvq_path.write_text(broken, encoding="utf-8")
            results = apply_all(_module.PATCHES, root)
            self.assertFalse(all(r.ok for r in results))
            self.assertTrue(any(e.edit_id == "prbe115-gate" for r in results for e in r.failed))
            self.assertEqual(broken, mmvq_path.read_text(encoding="utf-8"))

    def test_edit_contracts_are_fail_closed(self):
        for file_patch in _module.PATCHES:
            for edit in file_patch.edits:
                self.assertEqual(1, edit.expect_matches, edit.id)
                self.assertTrue(edit.guard, edit.id)
                self.assertTrue(edit.rationale, edit.id)


if __name__ == "__main__":
    unittest.main()
