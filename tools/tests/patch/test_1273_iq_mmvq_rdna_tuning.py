"""Offline mechanics tests for 1273_iq_mmvq_rdna_tuning."""

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
_PATCH_FILE = _REPO / "patches/1273_iq_mmvq_rdna_tuning/patch.py"
_VENDOR = _REPO / "tools/lab/iq-mmvq/vendor-b11233"

_spec = importlib.util.spec_from_file_location("patch_1273", _PATCH_FILE)
assert _spec is not None and _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


class Patch1273Mechanics(unittest.TestCase):
    def _tree(self):
        td = tempfile.TemporaryDirectory()
        root = Path(td.name)
        cuda = root / "ggml/src/ggml-cuda"
        cuda.mkdir(parents=True)
        for name in ("mmvq.cu", "mmvq.cuh", "vecdotq.cuh"):
            shutil.copy2(_VENDOR / name, cuda / name)
        return td, root, cuda / "mmvq.cu", cuda / "vecdotq.cuh"

    def test_iq4_xs_vdr2_halves_reuse_the_pristine_group_scale_and_lanes(self):
        # Mirror of the index math: pristine VDR=4 visits iqs=4k with words iqs+0..3 and q8 lanes
        # 0..3/4..7 under one 6-bit scale; VDR=2 visits iqs=4k and 4k+2 and must cover the same
        # words/lanes with that same scale (GPT review req_3c536b57ab6a4865: CRITICAL).
        def scale_bits(i):
            return (i // 8, i & 0x04, i // 2)  # scales_l index, scales_l shift, scales_h shift

        for iqs in range(0, 32, 4):
            pristine = {(iqs + j, j) for j in range(4)} | {(iqs + j, j + 4) for j in range(4)}
            halves = set()
            for half in (iqs, iqs + 2):
                q8_base = half & 0x02
                s_iqs = half & ~0x02
                self.assertEqual(scale_bits(s_iqs), scale_bits(iqs))
                halves |= {(half + j, q8_base + j) for j in range(2)} | {(half + j, q8_base + j + 4) for j in range(2)}
            self.assertEqual(halves, pristine)
        td, root, _, vecdot_path = self._tree()
        with td:
            self.assertTrue(all(r.ok for r in apply_all(_module.PATCHES, root)))
            vecdot = vecdot_path.read_text(encoding="utf-8")
            vdr2 = vecdot.split("vec_dot_iq4_xs_q8_1_vdr2(", 1)[1].split("\n}\n", 1)[0]
            self.assertIn("const int s_iqs = iqs & ~0x02;", vdr2)
            self.assertIn("bq4->scales_h >> (s_iqs/2)", vdr2)
            self.assertNotIn("scales_h >> (iqs/2)", vdr2)

    def test_apply_variants_and_idempotent(self):
        td, root, mmvq_path, vecdot_path = self._tree()
        with td:
            results = apply_all(_module.PATCHES, root)
            self.assertTrue(all(r.ok for r in results), [e.detail for r in results for e in r.failed])

            mmvq = mmvq_path.read_text(encoding="utf-8")
            vecdot = vecdot_path.read_text(encoding="utf-8")

            self.assertIn('bigcherry_iq_mmvq_env_enabled("BIGCHERRY_IQ_MMVQ_VDR")', mmvq)
            self.assertIn('bigcherry_iq_mmvq_env_enabled("BIGCHERRY_IQ_MMVQ_NWARPS")', mmvq)
            self.assertIn('BIGCHERRY_PATCH_HIT patch=1273_iq_mmvq path=%s type=%s arch=%s', mmvq)
            self.assertIn('case MMVQ_PARAMETERS_RDNA3_0: return "gfx1100";', mmvq)
            self.assertIn('case MMVQ_PARAMETERS_RDNA4:   return "gfx1201";', mmvq)
            self.assertIn('case GGML_TYPE_IQ4_XS:  return "iq4_xs";', mmvq)
            self.assertIn('case GGML_TYPE_IQ3_XXS: return "iq3_xxs";', mmvq)

            # Independent launch arms and the combined arm are explicit.
            self.assertIn('return "vdr_nwarps";', mmvq)
            self.assertIn('return tune_vdr ? "vdr" : "nwarps";', mmvq)
            self.assertIn('std::integral_constant<int, 2>{}, std::integral_constant<int, 1>{}', mmvq)
            self.assertIn('std::integral_constant<int, 4>{}, std::integral_constant<int, 2>{}', mmvq)
            self.assertIn('std::integral_constant<int, 2>{}, std::integral_constant<int, 8>{}', mmvq)
            self.assertIn('std::integral_constant<int, 4>{}, std::integral_constant<int, 4>{}', mmvq)
            self.assertIn('std::integral_constant<int, 1>{}, std::integral_constant<int, 2>{}', mmvq)
            self.assertIn('std::integral_constant<int, 1>{}, std::integral_constant<int, 1>{}', mmvq)
            self.assertIn('std::integral_constant<int, 2>{}, std::integral_constant<int, 2>{}', mmvq)

            # Lower-VDR helpers preserve the pristine functions and only add
            # the split variants selected by the tuned template instantiation.
            self.assertIn('vec_dot_iq3_xxs_q8_1_vdr1(', vecdot)
            self.assertIn('const int q8_base = 4 * (iqs & 1);', vecdot)
            self.assertIn('const int sign_base = 14 * (iqs & 1);', vecdot)
            self.assertIn('vec_dot_iq4_xs_q8_1_vdr2(', vecdot)
            self.assertIn('const int q8_base = iqs & 0x02;', vecdot)
            self.assertIn('static __device__ __forceinline__ float vec_dot_iq3_xxs_q8_1(', vecdot)
            self.assertIn('static __device__ __forceinline__ float vec_dot_iq4_xs_q8_1(', vecdot)

            before_mmvq = mmvq
            before_vecdot = vecdot
            second = apply_all(_module.PATCHES, root)
            self.assertTrue(all(r.ok for r in second), [e.detail for r in second for e in r.failed])
            self.assertEqual(before_mmvq, mmvq_path.read_text(encoding="utf-8"))
            self.assertEqual(before_vecdot, vecdot_path.read_text(encoding="utf-8"))

    def test_edit_contracts_are_fail_closed(self):
        for file_patch in _module.PATCHES:
            self.assertEqual("none", file_patch.language)
            for edit in file_patch.edits:
                self.assertEqual(1, edit.expect_matches, edit.id)
                self.assertTrue(edit.guard, edit.id)
                self.assertTrue(edit.rationale, edit.id)

    def test_anchor_mutation_fails_closed_without_writes(self):
        td, root, mmvq_path, vecdot_path = self._tree()
        with td:
            pristine_mmvq = mmvq_path.read_text(encoding="utf-8")
            pristine_vecdot = vecdot_path.read_text(encoding="utf-8")
            broken = pristine_mmvq.replace(
                "template <ggml_type type, int ncols_dst, bool has_fusion, bool small_k = false, bool halve_iters = false>\n",
                "template <ggml_type type, int ncols_dst, bool has_fusion, bool small_k = false, bool halve_iters = true>\n",
                1,
            )
            self.assertNotEqual(pristine_mmvq, broken)
            mmvq_path.write_text(broken, encoding="utf-8")

            results = apply_all(_module.PATCHES, root)
            self.assertFalse(all(r.ok for r in results))
            failures = [e for r in results for e in r.failed]
            self.assertTrue(any(e.edit_id == "iq-mmvq-kernel-template" for e in failures))
            self.assertEqual(broken, mmvq_path.read_text(encoding="utf-8"))
            self.assertEqual(pristine_vecdot, vecdot_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
