"""Offline mechanics tests for 1275_ar_small_latency."""

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
_PATCH_FILE = _REPO / "patches/1275_ar_small_latency/patch.py"
_VENDOR = _REPO / "tools/lab/allreduce-wire/vendor-b11233"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_module = _load("patch_1275", _PATCH_FILE)


class Patch1275Mechanics(unittest.TestCase):
    def _tree(self):
        td = tempfile.TemporaryDirectory()
        root = Path(td.name)
        cuda = root / "ggml/src/ggml-cuda"
        cuda.mkdir(parents=True)
        shutil.copy2(_VENDOR / "allreduce.cu", cuda / "allreduce.cu")
        shutil.copy2(_VENDOR / "allreduce.cuh", cuda / "allreduce.cuh")
        return td, root, cuda / "allreduce.cu"

    def test_apply_switches_marker_geometry_and_idempotent(self):
        td, root, path = self._tree()
        with td:
            results = apply_all(_module.PATCHES, root)
            self.assertTrue(all(r.ok for r in results), [e.detail for r in results for e in r.failed])
            text = path.read_text(encoding="utf-8")

            self.assertIn('getenv("BIGCHERRY_AR_SLOT_SYNC")', text)
            self.assertIn('strcmp(value, "host") == 0', text)
            self.assertIn('strcmp(value, "none") == 0', text)
            self.assertIn('ggml_cuda_ar_env_u64("BIGCHERRY_AR_SMALL_BLOCKS", 8)', text)
            self.assertIn('value == 1 || value == 2 || value == 4 || value == 8', text)
            self.assertIn('ggml_cuda_ar_env_u64("BIGCHERRY_AR_SMALL_THREADS", 256)', text)
            self.assertIn('value == 128 || value == 256', text)
            self.assertIn('BIGCHERRY_PATCH_HIT patch=1275_ar_small path=small_ar', text)
            self.assertIn('n_devices=%d blocks=%d threads=%d slot_sync=%s', text)
            self.assertIn('static std::atomic_flag logged = ATOMIC_FLAG_INIT;', text)

            self.assertIn('pool_lapped && !skip_host_sync', text)
            self.assertIn('single_chunk_small && p->slot_sync == ggml_cuda_ar_slot_sync::none', text)
            self.assertIn('ggml_cuda_ar_acquire_slot(p, skip_host_sync)', text)
            self.assertIn('const int slot = ggml_cuda_ar_acquire_slot(p).slot;', text)

            self.assertIn('dim3(p->small_blocks), dim3(p->small_threads)', text)
            self.assertIn('GGML_CUDA_AR_KERNEL_BLOCKS * GGML_CUDA_AR_ARRIVAL_STRIDE', text)
            self.assertIn('GGML_CUDA_AR_KERNEL_BLOCKS * GGML_CUDA_AR_ARRIVAL_STRIDE;', text)

            before = text
            second = apply_all(_module.PATCHES, root)
            self.assertTrue(all(r.ok for r in second), [e.detail for r in second for e in r.failed])
            self.assertEqual(before, path.read_text(encoding="utf-8"))

    def test_edit_contracts_are_fail_closed(self):
        for file_patch in _module.PATCHES:
            self.assertEqual("none", file_patch.language)
            for edit in file_patch.edits:
                self.assertEqual(1, edit.expect_matches, edit.id)
                self.assertTrue(edit.guard, edit.id)
                self.assertTrue(edit.rationale, edit.id)

    def test_anchor_mutation_fails_closed(self):
        td, root, path = self._tree()
        with td:
            pristine = path.read_text(encoding="utf-8")
            broken = pristine.replace(
                "struct ggml_cuda_ar_event_slot {\n    cudaEvent_t app = nullptr;  // upstream computation complete\n",
                "struct ggml_cuda_ar_event_slot_mutated {\n    cudaEvent_t app = nullptr;  // upstream computation complete\n",
                1,
            )
            self.assertNotEqual(pristine, broken)
            path.write_text(broken, encoding="utf-8")

            results = apply_all(_module.PATCHES, root)
            self.assertFalse(all(r.ok for r in results))
            failures = [e for r in results for e in r.failed]
            self.assertTrue(any(e.edit_id == "ar-small-slot-sync-type" for e in failures))
            self.assertEqual(broken, path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
