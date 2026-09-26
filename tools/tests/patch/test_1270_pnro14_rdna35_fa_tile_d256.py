"""Mechanics tests for 1270_pnro14_rdna35_fa_tile_d256."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bigcherry.patcher import apply_all  # noqa: E402

_REPO = Path(__file__).resolve().parents[3]
_PATCH_FILE = _REPO / "patches/1270_pnro14_rdna35_fa_tile_d256/patch.py"
_spec = importlib.util.spec_from_file_location("patch_1270", _PATCH_FILE)
assert _spec is not None and _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

_SOURCE = r'''#include "common.cuh"
#include "fattn-common.cuh"

#define GGML_CUDA_FATTN_TILE_CONFIG_CASE(DKQ_, DV_, ncols_, nthreads, occupancy, nbatch_fa, nbatch_K) \
    if (DKQ == (DKQ_) && DV == (DV_) && ncols == (ncols_)) { return 1; }

static constexpr __host__ __device__ uint32_t ggml_cuda_fattn_tile_get_config_amd_rdna(
        const int DKQ, const int DV, const int ncols) {
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256, 32, 256, 3, 64, 128)
    return 0;
}

static constexpr __host__ __device__ uint32_t ggml_cuda_fattn_tile_get_config_amd(
        const int DKQ, const int DV, const int ncols) {
    return 0;
}

static __host__ uint32_t ggml_cuda_fattn_tile_get_config(const int DKQ, const int DV, const int ncols, const int cc) {
    if (GGML_CUDA_CC_IS_AMD(cc)) {
        if (GGML_CUDA_CC_IS_RDNA(cc)) {
            return ggml_cuda_fattn_tile_get_config_amd_rdna(DKQ, DV, ncols);
        }
        return ggml_cuda_fattn_tile_get_config_amd(DKQ, DV, ncols);
    }
    if (fast_fp16_available(cc)) {
        return ggml_cuda_fattn_tile_get_config_nvidia_fp16(DKQ, DV, ncols);
    }
    return ggml_cuda_fattn_tile_get_config_nvidia_fp32(DKQ, DV, ncols);
}

static constexpr __device__ uint32_t ggml_cuda_fattn_tile_get_config(const int DKQ, const int DV, const int ncols) {
#ifdef GGML_USE_HIP
#ifdef RDNA
    return ggml_cuda_fattn_tile_get_config_amd_rdna(DKQ, DV, ncols);
#else
    return ggml_cuda_fattn_tile_get_config_amd(DKQ, DV, ncols);
#endif // RDNA
#else
#ifdef FAST_FP16_AVAILABLE
    return ggml_cuda_fattn_tile_get_config_nvidia_fp16(DKQ, DV, ncols);
#else
    return ggml_cuda_fattn_tile_get_config_nvidia_fp32(DKQ, DV, ncols);
#endif // FAST_FP16_AVAILABLE
#endif // GGML_USE_HIP
}
'''


class Patch1270Mechanics(unittest.TestCase):
    def _tree(self):
        td = tempfile.TemporaryDirectory()
        root = Path(td.name)
        path = root / "ggml/src/ggml-cuda/fattn-tile.cuh"
        path.parent.mkdir(parents=True)
        path.write_text(_SOURCE, encoding="utf-8")
        return td, root, path

    def test_apply_and_idempotent(self):
        td, root, path = self._tree()
        with td:
            first = apply_all(_module.PATCHES, root)
            self.assertTrue(all(r.ok for r in first), [e.detail for r in first for e in r.failed])
            text = path.read_text()
            self.assertIn("GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256, 32, 256, 4, 64, 64)", text)
            self.assertIn("GGML_CUDA_CC_IS_RDNA3_5(cc)", text)
            self.assertIn("#if defined(RDNA3_5)", text)
            self.assertIn("bigcherry_pnro14_logged", text)
            before = text
            second = apply_all(_module.PATCHES, root)
            self.assertTrue(all(r.ok for r in second))
            self.assertEqual(before, path.read_text())

    def test_missing_host_selector_fails_closed(self):
        td, root, path = self._tree()
        with td:
            broken = _SOURCE.replace("static __host__ uint32_t ggml_cuda_fattn_tile_get_config", "static __host__ uint32_t wrong_selector", 1)
            path.write_text(broken, encoding="utf-8")
            results = apply_all(_module.PATCHES, root)
            self.assertFalse(all(r.ok for r in results))


if __name__ == "__main__":
    unittest.main()
