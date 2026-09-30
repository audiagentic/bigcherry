"""Offline mechanics tests for 1272_ar_host_compressed_wire."""

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
_PATCH_FILE = _REPO / "patches/1272_ar_host_compressed_wire/patch.py"
_P2P_PATCH_FILE = _REPO / "patches/1252_nro03_allreduce_p2p_provider/patch.py"
_VENDOR = _REPO / "tools/lab/allreduce-wire/vendor-b11233"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_module = _load("patch_1272", _PATCH_FILE)
_p2p_module = _load("patch_1252_for_1272", _P2P_PATCH_FILE)


def _allreduce_only(module):
    return [p for p in module.PATCHES if p.path == "ggml/src/ggml-cuda/allreduce.cu"]


class Patch1272Mechanics(unittest.TestCase):
    def _tree(self):
        td = tempfile.TemporaryDirectory()
        root = Path(td.name)
        cuda = root / "ggml/src/ggml-cuda"
        cuda.mkdir(parents=True)
        shutil.copy2(_VENDOR / "allreduce.cu", cuda / "allreduce.cu")
        shutil.copy2(_VENDOR / "allreduce.cuh", cuda / "allreduce.cuh")
        return td, root, cuda / "allreduce.cu"

    def test_apply_branches_and_idempotent(self):
        td, root, path = self._tree()
        with td:
            results = apply_all(_module.PATCHES, root)
            self.assertTrue(all(r.ok for r in results), [e.detail for r in results for e in r.failed])
            text = path.read_text(encoding="utf-8")

            self.assertIn('if (value == nullptr || value[0] == \'\\0\')', text)
            self.assertIn('return ggml_cuda_ar_wire_override::pristine;', text)
            self.assertIn('if (p->wire_override != ggml_cuda_ar_wire_override::pristine)', text)
            self.assertIn('p->bf16_threshold   = ggml_cuda_ar_env_u64("GGML_CUDA_AR_BF16_THRESHOLD", 1);', text)
            self.assertIn('const bool use_bf16 =', text)

            self.assertIn('BIGCHERRY_PATCH_HIT patch=1272_ar_wire path=ar_wire_%s', text)
            self.assertIn('static std::atomic_flag logged = ATOMIC_FLAG_INIT;', text)
            for wire in ("f32", "bf16", "f16", "q8_0"):
                self.assertIn(f'strcmp(value, "{wire}") == 0', text)

            self.assertIn('case ggml_cuda_ar_wire_override::f32:  return ggml_cuda_ar_allreduce_wire_typed<T_dst, float>', text)
            self.assertIn('case ggml_cuda_ar_wire_override::bf16: return ggml_cuda_ar_allreduce_wire_typed<T_dst, nv_bfloat16>', text)
            self.assertIn('case ggml_cuda_ar_wire_override::f16:  return ggml_cuda_ar_allreduce_wire_typed<T_dst, half>', text)
            self.assertIn('case ggml_cuda_ar_wire_override::q8_0: return ggml_cuda_ar_allreduce_wire_q8<T_dst>', text)

            self.assertIn('return ggml_cuda_ar_allreduce_copy_outer<T_wire, T_dst>', text)
            self.assertIn('ggml_cuda_ar_kernel<T_dst, T_wire><<<', text)
            self.assertIn('ggml_cuda_cast<float>(d_low) + ggml_cuda_cast<float>(src[i])', text)
            self.assertIn('ggml_cuda_cast<float>(d_low) + ggml_cuda_cast<float>(wire[k])', text)

            before = text
            second = apply_all(_module.PATCHES, root)
            self.assertTrue(all(r.ok for r in second), [e.detail for r in second for e in r.failed])
            self.assertEqual(before, path.read_text(encoding="utf-8"))

    def test_q8_0_codec_and_shared_finish_cover_both_paths(self):
        td, root, path = self._tree()
        with td:
            results = apply_all(_module.PATCHES, root)
            self.assertTrue(all(r.ok for r in results), [e.detail for r in results for e in r.failed])
            text = path.read_text(encoding="utf-8")

            self.assertEqual(1, text.count('static __global__ void ggml_cuda_ar_quantize_q8_0_kernel('))
            self.assertEqual(1, text.count('static __global__ void ggml_cuda_ar_q8_0_add_kernel('))
            self.assertIn('block_q8_0  * __restrict__ dst', text)
            self.assertIn('const int lane = threadIdx.x % QK8_0;', text)
            self.assertIn('warp_reduce_max<QK8_0>(fabsf(x))', text)
            self.assertIn('dst[ib].d = ggml_cuda_cast<half>(d);', text)
            self.assertIn('const float a = ggml_cuda_cast<float>(local[ib].d)', text)
            self.assertIn('dst[i] = ggml_cuda_cast<T_dst>(a + b);', text)

            self.assertIn('static __global__ void ggml_cuda_ar_q8_0_add_residual_kernel(', text)
            self.assertIn('static void ggml_cuda_ar_launch_finish(', text)
            self.assertIn('residual_buf ? residual_buf[i] : nullptr', text)

            self.assertIn('static bool ggml_cuda_ar_allreduce_copy_q8_impl(', text)
            self.assertIn('return ggml_cuda_ar_allreduce_copy_q8_outer<T_dst>', text)
            self.assertIn('reinterpret_cast<const block_q8_0 *>(p->dev_tmp[i])', text)

            self.assertIn('static __global__ void ggml_cuda_ar_q8_0_mapped_kernel(', text)
            self.assertIn('p->host_buf[i].dev + (size_t) slot * p->buf_bytes', text)
            self.assertIn('p->host_buf[peer].dev + (size_t) slot * p->buf_bytes', text)
            self.assertIn('ggml_cuda_ar_arrival_ptr(p, slot, peer)', text)

    def test_composes_after_1252_p2p_provider(self):
        td, root, path = self._tree()
        with td:
            p2p = apply_all(_allreduce_only(_p2p_module), root)
            self.assertTrue(all(r.ok for r in p2p), [e.detail for r in p2p for e in r.failed])
            wire = apply_all(_module.PATCHES, root)
            self.assertTrue(all(r.ok for r in wire), [e.detail for r in wire for e in r.failed])
            text = path.read_text(encoding="utf-8")
            self.assertIn('p2p_enabled;', text)
            self.assertIn('ggml_cuda_ar_wire_override wire_override;', text)
            self.assertEqual(1, text.count('ggml_cuda_ar_wire_override wire_override;'))

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
            self.assertTrue(any(e.edit_id == "ar-wire-enum-parser-trace" for e in failures))
            self.assertEqual(broken, path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
