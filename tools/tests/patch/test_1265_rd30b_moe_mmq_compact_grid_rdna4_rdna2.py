"""Mechanics tests for 1265_rd30b_moe_mmq_compact_grid_rdna4_rdna2.

GPT code review 2026-09-27 (req_6c90e1ebba83464a): the first version of this
gate used GGML_CUDA_CC_IS_RDNA4(cc)/IS_RDNA2(cc) -- open family ranges that
admit untested siblings (IS_RDNA4 has no upper bound at all). Only gfx1201
and gfx1030 were ever measured, so these tests fail closed against the
family-macro regression, not just against a missing anchor.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.patcher import apply_all  # noqa: E402

_REPO = Path(__file__).resolve().parents[3]
_PATCH_FILE = _REPO / "patches/1265_rd30b_moe_mmq_compact_grid_rdna4_rdna2/patch.py"
_spec = importlib.util.spec_from_file_location("patch_1265_rd30b", _PATCH_FILE)
assert _spec is not None and _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

# What 1237 (a required predecessor, not composed here) leaves in place --
# this patch's own anchor target.
_MMQ_CUH = """static bool mmq_moe_compact_supported(const int cc) {
    return cc == GGML_CUDA_CC_RDNA3;
}
"""


class Patch1265Mechanics(unittest.TestCase):
    def _tree(self) -> tuple[tempfile.TemporaryDirectory, Path]:
        tmp = tempfile.TemporaryDirectory()
        root = Path(tmp.name)
        (root / "ggml/src/ggml-cuda").mkdir(parents=True)
        (root / "ggml/src/ggml-cuda/mmq.cuh").write_text(_MMQ_CUH, encoding="utf-8")
        return tmp, root

    def test_apply_and_idempotent(self) -> None:
        tmp, root = self._tree()
        with tmp:
            first = apply_all(_module.PATCHES, root)
            self.assertTrue(all(r.ok for r in first), [e.detail for r in first for e in r.failed])
            text = (root / "ggml/src/ggml-cuda/mmq.cuh").read_text(encoding="utf-8")
            # Exact-match only: no open family macro (IS_RDNA4/IS_RDNA2) that
            # would admit an untested sibling architecture.
            self.assertNotIn("GGML_CUDA_CC_IS_RDNA4", text)
            self.assertNotIn("GGML_CUDA_CC_IS_RDNA2", text)
            self.assertIn("GGML_CUDA_CC_RDNA3", text)
            self.assertIn("GGML_CUDA_CC_OFFSET_AMD + 0x1201", text)  # gfx1201 exactly
            self.assertIn("GGML_CUDA_CC_RDNA2", text)  # == gfx1030 exactly (its own threshold)

            second = apply_all(_module.PATCHES, root)
            self.assertTrue(all(r.ok for r in second), [e.detail for r in second for e in r.failed])
            self.assertEqual((root / "ggml/src/ggml-cuda/mmq.cuh").read_text(encoding="utf-8"), text)

    def test_missing_anchor_fails_closed(self) -> None:
        tmp, root = self._tree()
        with tmp:
            (root / "ggml/src/ggml-cuda/mmq.cuh").write_text(
                "static bool mmq_moe_compact_supported(const int cc) {\n"
                "    return cc == GGML_CUDA_CC_RDNA4;\n"  # already widened by someone else
                "}\n",
                encoding="utf-8",
            )
            result = apply_all(_module.PATCHES, root)
            self.assertFalse(all(r.ok for r in result))


if __name__ == "__main__":
    unittest.main()
