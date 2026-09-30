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
_VENDOR = _REPO / "tools/lab/allreduce-wire/vendor-b11233"

_spec = importlib.util.spec_from_file_location("patch_1272", _PATCH_FILE)
assert _spec is not None and _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


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

            # Unset env remains on the pristine b11233 path; explicit values
            # alone enter 1272.
            self.assertIn('if (value == nullptr || value[0] == \'\\0\')', text)
            self.assertIn('return ggml_cuda_ar_wire_override::pristine;', text)
            self.assertIn('if (p->wire_override != ggml_cuda_ar_wire_override::pristine)', text)
            self.assertIn('p->bf16_threshold   = ggml_cuda_ar_env_u64("GGML_CUDA_AR_BF16_THRESHOLD", 1);', text)
            self.assertIn('const bool use_bf16 =', text)

            # Once-per-process activation marker and all implemented explicit
            # wire selectors.
            self.assertIn('BIGCHERRY_PATCH_HIT patch=1272_ar_wire path=ar_wire_%s', text)
            self.assertIn('static std::atomic_flag logged = ATOMIC_FLAG_INIT;', text)
            self.assertIn('strcmp(value, "f32") == 0', text)
            self.assertIn('strcmp(value, "bf16") == 0', text)
            self.assertIn('strcmp(value, "f16") == 0', text)
            self.assertIn('case ggml_cuda_ar_wire_override::f32:  return ggml_cuda_ar_allreduce_wire_typed<T_dst, float>', text)
            self.assertIn('case ggml_cuda_ar_wire_override::bf16: return ggml_cuda_ar_allreduce_wire_typed<T_dst, nv_bfloat16>', text)
            self.assertIn('case ggml_cuda_ar_wire_override::f16:  return ggml_cuda_ar_allreduce_wire_typed<T_dst, half>', text)

            # Same explicit T_wire drives both provider-preserving paths:
            # copy-engine for large wire payloads and mapped-host chunked
            # kernels for decode-sized payloads. Receiver accumulation stays
            # fused through the pristine F32-cast add/kernel machinery.
            self.assertIn('const bool use_copy_engine =', text)
            self.assertIn('return ggml_cuda_ar_allreduce_copy_outer<T_wire, T_dst>(', text)
            self.assertIn('ggml_cuda_ar_kernel<T_dst, T_wire><<<', text)
            self.assertIn('ggml_cuda_cast<float>(d_low) + ggml_cuda_cast<float>(src[i])', text)
            self.assertIn('ggml_cuda_cast<float>(d_low) + ggml_cuda_cast<float>(wire[k])', text)

            # q8_0 is deliberately fail-closed in this phase rather than
            # silently becoming pristine/native wire.
            self.assertIn('ggml_cuda_ar_wire_override::q8_0_unsupported', text)
            self.assertIn('GGML_CUDA_AR_WIRE=q8_0 is not implemented by patch 1272 phase 1', text)

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
            self.assertTrue(any(e.edit_id == "ar-wire-enum-parser-trace" for e in failures))
            self.assertEqual(broken, path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
