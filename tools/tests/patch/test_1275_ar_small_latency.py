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
_ROOT3_PATCH_FILE = _REPO / "patches/1244_gp11_internal_allreduce_nway_root/patch.py"
_VENDOR = _REPO / "tools/lab/allreduce-wire/vendor-b11233"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_module = _load("patch_1275", _PATCH_FILE)
_root3_module = _load("patch_1244_for_1275", _ROOT3_PATCH_FILE)


def _allreduce_only(module):
    return [p for p in module.PATCHES if p.path == "ggml/src/ggml-cuda/allreduce.cu"]


class Patch1275Mechanics(unittest.TestCase):
    def _tree(self):
        td = tempfile.TemporaryDirectory()
        root = Path(td.name)
        cuda = root / "ggml/src/ggml-cuda"
        cuda.mkdir(parents=True)
        shutil.copy2(_VENDOR / "allreduce.cu", cuda / "allreduce.cu")
        return td, root, cuda / "allreduce.cu"

    def test_pristine_apply_defaults_controls_trace_and_idempotent(self):
        td, root, path = self._tree()
        with td:
            results = apply_all(_module.PATCHES, root)
            self.assertTrue(all(r.ok for r in results), [e.detail for r in results for e in r.failed])
            text = path.read_text(encoding="utf-8")

            self.assertIn('getenv("BIGCHERRY_AR_SLOT_SYNC")', text)
            self.assertIn('"BIGCHERRY_AR_SMALL_BLOCKS", 8', text)
            self.assertIn('"BIGCHERRY_AR_SMALL_THREADS", 256', text)
            self.assertIn('ggml_cuda_ar_slot_sync::host', text)
            self.assertIn('ggml_cuda_ar_slot_sync::stream', text)
            self.assertIn('ggml_cuda_ar_slot_sync::none', text)

            self.assertIn('const bool host_wait = !single_chunk_small || p->slot_sync == ggml_cuda_ar_slot_sync::host;', text)
            self.assertIn('cudaStreamWaitEvent(streams[i], p->ev_pool[i][slot].ker)', text)
            self.assertIn('p->call_count <= GGML_CUDA_AR_POOL_SIZE', text)
            self.assertIn('const bool single_chunk_small = ne <= (int64_t) max_chunk_elems;', text)

            self.assertIn('dim3(single_chunk_small ? p->small_blocks : GGML_CUDA_AR_KERNEL_BLOCKS)', text)
            self.assertIn('dim3(single_chunk_small ? p->small_threads : 256)', text)
            self.assertIn('GGML_CUDA_AR_KERNEL_BLOCKS * GGML_CUDA_AR_ARRIVAL_STRIDE', text)

            self.assertIn('host_enqueue_us=%llu slot_wait_us=%llu', text)
            self.assertIn('BIGCHERRY_PATCH_HIT patch=1275_ar_small path=small_ar n_devices=%d blocks=%d threads=%d slot_sync=%s.', text)
            self.assertIn('static std::atomic_flag logged = ATOMIC_FLAG_INIT;', text)

            before = text
            second = apply_all(_module.PATCHES, root)
            self.assertTrue(all(r.ok for r in second), [e.detail for r in second for e in r.failed])
            self.assertEqual(before, path.read_text(encoding="utf-8"))

    def test_composes_after_1244_root3(self):
        td, root, path = self._tree()
        with td:
            root3 = apply_all(_allreduce_only(_root3_module), root)
            self.assertTrue(all(r.ok for r in root3), [e.detail for r in root3 for e in r.failed])
            latency = apply_all(_module.PATCHES, root)
            self.assertTrue(all(r.ok for r in latency), [e.detail for r in latency for e in r.failed])
            text = path.read_text(encoding="utf-8")

            self.assertIn('static bool ggml_cuda_ar_allreduce_root3(', text)
            self.assertIn('ggml_cuda_ar_stream_wait_old_slot(p, slot, streams, single_chunk_small);', text)
            self.assertIn('ggml_cuda_ar_kernel3<<<dim3(single_chunk_small ? p->small_blocks : GGML_CUDA_AR_KERNEL_BLOCKS)', text)
            self.assertEqual(2, text.count('ggml_cuda_ar_kernel3_leaf<<<dim3(single_chunk_small ? p->small_blocks : GGML_CUDA_AR_KERNEL_BLOCKS)'))
            self.assertIn('ggml_cuda_ar_arrival_ptr3(p, 0, slot, 1)', text)

    def test_root3_edits_are_not_applicable_on_pristine(self):
        td, root, _ = self._tree()
        with td:
            results = apply_all(_module.PATCHES, root)
            statuses = {
                e.edit_id: e.status
                for r in results
                for e in r.results
                if e.edit_id.startswith("ar-small-root3-")
            }
            self.assertTrue(statuses)
            self.assertTrue(all(status == "not-applicable" for status in statuses.values()), statuses)

    def test_edit_contracts_are_fail_closed(self):
        for file_patch in _module.PATCHES:
            self.assertEqual("none", file_patch.language)
            for edit in file_patch.edits:
                self.assertGreaterEqual(edit.expect_matches, 1, edit.id)
                self.assertTrue(edit.guard, edit.id)
                self.assertTrue(edit.rationale, edit.id)

    def test_anchor_mutation_fails_closed(self):
        td, root, path = self._tree()
        with td:
            pristine = path.read_text(encoding="utf-8")
            broken = pristine.replace(
                "struct ggml_cuda_ar_pipeline {\n",
                "struct ggml_cuda_ar_pipeline_mutated {\n",
                1,
            )
            self.assertNotEqual(pristine, broken)
            path.write_text(broken, encoding="utf-8")

            results = apply_all(_module.PATCHES, root)
            self.assertFalse(all(r.ok for r in results))
            failures = [e for r in results for e in r.failed]
            self.assertTrue(any(e.edit_id == "ar-small-support" for e in failures))
            self.assertEqual(broken, path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
