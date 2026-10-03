"""Mechanics tests for 1266_rd05_wmma_fa_tileq_sync."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.patcher import apply_all  # noqa: E402

_REPO = Path(__file__).resolve().parents[3]
_PATCH_FILE = _REPO / "patches/1266_rd05_wmma_fa_tileq_sync/patch.py"
_spec = importlib.util.spec_from_file_location("patch_1266_rd05", _PATCH_FILE)
assert _spec is not None and _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

_FATTN_MMA = """void f() {
        if (np > 1) {
            __syncthreads();
        }
        work();
        }

        kbc += iter_k;
        kbc -= kbc % iter_k;
}
"""

_FATTN = """#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-mma-f16.cuh"
#include "fattn-tile.cuh"
#include "fattn-vec.cuh"
#include "fattn.cuh"

void launch() {
        case BEST_FATTN_KERNEL_MMA_F16:
            ggml_cuda_flash_attn_ext_mma_f16(ctx, dst);
            break;
}
"""


class Patch1266Mechanics(unittest.TestCase):
    def _tree(self) -> tuple[tempfile.TemporaryDirectory, Path]:
        td = tempfile.TemporaryDirectory()
        root = Path(td.name)
        base = root / "ggml/src/ggml-cuda"
        base.mkdir(parents=True)
        (base / "fattn-mma-f16.cuh").write_text(_FATTN_MMA, encoding="utf-8")
        (base / "fattn.cu").write_text(_FATTN, encoding="utf-8")
        return td, root

    def test_apply_and_idempotent(self):
        td, root = self._tree()
        with td:
            first = apply_all(_module.PATCHES, root)
            self.assertTrue(all(result.ok for result in first), [e.detail for r in first for e in r.failed])
            text = (root / "ggml/src/ggml-cuda/fattn-mma-f16.cuh").read_text(encoding="utf-8")
            self.assertIn("k00 + nbatch_combine < DV/2", text)
            self.assertIn("reuses the tile_Q buffer", text)
            dispatch = (root / "ggml/src/ggml-cuda/fattn.cu").read_text(encoding="utf-8")
            self.assertIn("patch=1266_rd05_wmma_fa_tileq_sync", dispatch)
            before = {p: (root / p).read_text(encoding="utf-8") for p in (
                "ggml/src/ggml-cuda/fattn-mma-f16.cuh", "ggml/src/ggml-cuda/fattn.cu")}
            second = apply_all(_module.PATCHES, root)
            self.assertTrue(all(result.ok for result in second))
            self.assertEqual(before, {p: (root / p).read_text(encoding="utf-8") for p in before})

    def test_missing_anchor_fails_closed(self):
        td, root = self._tree()
        with td:
            path = root / "ggml/src/ggml-cuda/fattn.cu"
            path.write_text(_FATTN.replace("BEST_FATTN_KERNEL_MMA_F16", "BEST_FATTN_KERNEL_TILE_F32"), encoding="utf-8")
            before = (root / "ggml/src/ggml-cuda/fattn-mma-f16.cuh").read_text(encoding="utf-8")
            results = apply_all(_module.PATCHES, root)
            self.assertFalse(all(result.ok for result in results))
            self.assertEqual(before, (root / "ggml/src/ggml-cuda/fattn-mma-f16.cuh").read_text(encoding="utf-8"))

    def test_ambiguous_anchor_fails_closed(self):
        td, root = self._tree()
        with td:
            path = root / "ggml/src/ggml-cuda/fattn-mma-f16.cuh"
            duplicate = "        if (np > 1) {\n            __syncthreads();\n        }\n"
            path.write_text(_FATTN_MMA + duplicate, encoding="utf-8")
            results = apply_all(_module.PATCHES, root)
            self.assertFalse(all(result.ok for result in results))
            detail = "\n".join(e.detail for result in results for e in result.failed)
            self.assertIn("matched 2 time(s)", detail)


if __name__ == "__main__":
    unittest.main()
