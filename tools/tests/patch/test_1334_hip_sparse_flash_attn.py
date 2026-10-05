"""Offline mechanics tests for 1334_hip_sparse_flash_attn."""

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
_V = _REPO / "vendor/llama.cpp"
_FILES = ("ggml/src/ggml-cuda/fattn.cu", "ggml/src/ggml-cuda/fattn-mma-f16.cuh")


def _load():
    spec = importlib.util.spec_from_file_location("patch_1334", _REPO / "patches/1334_hip_sparse_flash_attn/patch.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_P = _load()


@unittest.skipUnless(all((_V / f).exists() for f in _FILES), "pinned vendor checkout not present")
class Patch1334Mechanics(unittest.TestCase):
    def _root(self, td):
        root = Path(td)
        for rel in _FILES:
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(_V / rel, root / rel)
        return root

    def test_apply_and_idempotent(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._root(td)
            res = apply_all(_P.PATCHES, root)
            self.assertTrue(all(r.ok for r in res), [e.detail for r in res for e in r.failed])
            fa = (root / _FILES[0]).read_text(encoding="utf-8")
            mma = (root / _FILES[1]).read_text(encoding="utf-8")
            # one HIP index kernel, defined before upstream's NVIDIA one, on the AMD wave primitives
            hip = fa[fa.index("#if defined(GGML_USE_HIP)\n#include <cstdlib>"):fa.index("#endif // defined(GGML_USE_HIP)\n\n#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)")]
            self.assertEqual(hip.count("static __global__ void flash_attn_mask_to_sparse_indices("), 1)
            self.assertIn("selected_wave[item] = __ballot(selected);", hip)
            self.assertIn("__popcll(selected_wave[item] & lane_mask)", hip)
            self.assertNotIn("__ballot_sync", hip)   # CUDA-only
            self.assertNotIn("WARP_SIZE", hip)       # the macro is 32; the wave width comes from warpSize
            self.assertIn("const int lane     = tid % warpSize;", hip)
            self.assertIn("indices[i] = -1;", hip)
            self.assertIn("counts_ptr[int64_t(sequence)*gridDim.x + group] = count;", hip)
            self.assertEqual(fa.count("static __global__ void flash_attn_mask_to_sparse_indices("), 2)
            # no HIP abort / early false left in the host functions
            self.assertNotIn("sparse flash attention is only supported on NVIDIA CUDA", fa)
            self.assertEqual(fa.count("#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)"), 0)
            # RDNA selection is opt-in and the NVIDIA term is unchanged
            self.assertIn("const bool bc_arch_ok = amd_wmma_available(cc) && bc_fa_sparse_enabled();", fa)
            self.assertIn("const bool bc_arch_ok = GGML_CUDA_CC_IS_NVIDIA(cc) && turing_mma_available(cc);", fa)
            self.assertIn('getenv("BIGCHERRY_FA_SPARSE") != nullptr && atoi(', fa)
            # RDNA picks ncols2 = 8 only when the sparse path is taken, before its exact-divisibility choices
            rdna = fa[fa.index("// On RDNA it is preferable to minimize wasted compute"):]
            self.assertLess(rdna.index("shall_use_sparse(cc, dst, 8, 8)"), rdna.index("if (use_gqa_opt && gqa_ratio % 8 == 0)"))
            # both dispatch sites compile for HIP
            self.assertIn("#if !defined(GGML_USE_MUSA)  // BigCherry 1334: compiled for HIP\n    if constexpr", fa)
            self.assertIn("#if !defined(GGML_USE_MUSA)  // BigCherry 1334: compiled for HIP\n        if constexpr", mma)
            second = apply_all(_P.PATCHES, root)
            self.assertTrue(all(r.ok for r in second), [e.detail for r in second for e in r.failed])
            self.assertEqual(fa, (root / _FILES[0]).read_text(encoding="utf-8"))

    def test_missing_guard_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._root(td)
            p = root / _FILES[0]
            p.write_text(p.read_text(encoding="utf-8").replace(
                "    return GGML_CUDA_CC_IS_NVIDIA(cc) && turing_mma_available(cc) &&\n",
                "    return GGML_CUDA_CC_IS_NVIDIA(cc) && mma_available(cc) &&\n"), encoding="utf-8")
            res = apply_all(_P.PATCHES, root)
            self.assertFalse(all(r.ok for r in res))


if __name__ == "__main__":
    unittest.main()
