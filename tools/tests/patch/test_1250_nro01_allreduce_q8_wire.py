"""Offline mechanics tests for 1250_nro01_allreduce_q8_wire after 1272 migration."""

from __future__ import annotations

import importlib.util
import shutil
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bigcherry.patcher import apply_all  # noqa: E402

_REPO = Path(__file__).resolve().parents[3]
_VENDOR = _REPO / "tools/lab/allreduce-wire/vendor-b11233"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_p1250 = _load("patch_1250", _REPO / "patches/1250_nro01_allreduce_q8_wire/patch.py")
_p1252 = _load("patch_1252_for_1250", _REPO / "patches/1252_nro03_allreduce_p2p_provider/patch.py")
_p1272 = _load("patch_1272_for_1250", _REPO / "patches/1272_ar_host_compressed_wire/patch.py")


def _allreduce_only(module):
    return [p for p in module.PATCHES if p.path == "ggml/src/ggml-cuda/allreduce.cu"]


class Patch1250SharedWireMechanics(unittest.TestCase):
    def _tree(self):
        td = tempfile.TemporaryDirectory()
        root = Path(td.name)
        cuda = root / "ggml/src/ggml-cuda"
        cuda.mkdir(parents=True)
        shutil.copy2(_VENDOR / "allreduce.cu", cuda / "allreduce.cu")
        return td, root, cuda / "allreduce.cu"

    def _apply_stack(self, root: Path):
        for module in (_p1252, _p1272):
            results = apply_all(_allreduce_only(module), root)
            self.assertTrue(all(r.ok for r in results), [e.detail for r in results for e in r.failed])
        results = apply_all(_allreduce_only(_p1250), root)
        self.assertTrue(all(r.ok for r in results), [e.detail for r in results for e in r.failed])
        return results

    def test_p2p_reuses_1272_codec_and_finish(self):
        td, root, path = self._tree()
        with td:
            self._apply_stack(root)
            text = path.read_text(encoding="utf-8")

            self.assertEqual(1, text.count("static __global__ void ggml_cuda_ar_quantize_q8_0_kernel("))
            self.assertEqual(1, text.count("static __global__ void ggml_cuda_ar_q8_0_add_kernel("))
            self.assertEqual(1, text.count("enum class ggml_cuda_ar_wire_override"))
            self.assertIn("static bool ggml_cuda_ar_allreduce_p2p_q8_outer(", text)
            self.assertIn("ggml_cuda_ar_allreduce_p2p_impl<block_q8_0, T_dst>", text)
            self.assertIn("ggml_cuda_ar_launch_finish(\n            src_buf[i], dst_buf[i]", text)
            self.assertIn("if (p->p2p_enabled) {\n            return ggml_cuda_ar_allreduce_p2p_q8_outer<T_dst>", text)
            self.assertIn("BIGCHERRY_PATCH_HIT patch=1250_nro01 path=allreduce_q8_0_p2p_shared", text)

    def test_fused_residual_reuses_shared_quantize_and_host_or_p2p_finish(self):
        td, root, path = self._tree()
        with td:
            self._apply_stack(root)
            text = path.read_text(encoding="utf-8")
            self.assertIn("bool ggml_cuda_ar_allreduce_fused_add(", text)
            self.assertIn("p->wire_override != ggml_cuda_ar_wire_override::q8_0", text)
            self.assertIn("ggml_cuda_ar_quantize_q8_0_kernel<float><<<", text)
            self.assertIn("return ggml_cuda_ar_allreduce_p2p_q8_outer<float>(p, backends, src, dst, residual, ne);", text)
            self.assertIn("return ggml_cuda_ar_allreduce_copy_q8_outer<float>(p, backends, src, dst, residual, ne);", text)
            self.assertIn("ggml_cuda_ar_q8_0_add_residual_kernel<T_dst><<<", text)

    def test_allreduce_overlay_is_idempotent(self):
        td, root, path = self._tree()
        with td:
            self._apply_stack(root)
            before = path.read_text(encoding="utf-8")
            second = apply_all(_allreduce_only(_p1250), root)
            self.assertTrue(all(r.ok for r in second), [e.detail for r in second for e in r.failed])
            self.assertEqual(before, path.read_text(encoding="utf-8"))

    def test_dependency_and_conflict_wiring(self):
        p1250 = tomllib.loads((_REPO / "patches/1250_nro01_allreduce_q8_wire/patch.toml").read_text(encoding="utf-8"))
        p1272 = tomllib.loads((_REPO / "patches/1272_ar_host_compressed_wire/patch.toml").read_text(encoding="utf-8"))
        self.assertEqual(["1252_nro03_allreduce_p2p_provider", "1272_ar_host_compressed_wire"], p1250["requires"])
        self.assertNotIn("1250_nro01_allreduce_q8_wire", p1272["conflicts"])

    def test_all_edits_fail_closed(self):
        for file_patch in _p1250.PATCHES:
            for edit in file_patch.edits:
                self.assertEqual(1, edit.expect_matches, edit.id)
                self.assertTrue(edit.guard, edit.id)
                self.assertTrue(edit.rationale, edit.id)


if __name__ == "__main__":
    unittest.main()
