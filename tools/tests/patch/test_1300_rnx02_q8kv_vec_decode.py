"""Mechanics tests for the default-off RNX02 Q8 vector selector package."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import sys

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))

from bigcherry.patcher import apply_all  # noqa: E402


def _load_patch():
    path = _REPO / "patches/1300_rnx02_q8kv_vec_decode/patch.py"
    spec = importlib.util.spec_from_file_location("patch_1300_rnx02", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RNX02PatchTests(unittest.TestCase):
    def test_metadata_and_edits_are_default_off_and_anchored(self) -> None:
        module = _load_patch()
        self.assertEqual(module.STATE, "untested")
        self.assertEqual(len(module.PATCHES), 1)
        self.assertEqual(module.PATCHES[0].path, "ggml/src/ggml-cuda/fattn.cu")
        self.assertEqual(
            [edit.expect_matches for edit in module.PATCHES[0].edits], [1, 1]
        )
        selector = module.PATCHES[0].edits[1].text
        self.assertIn("BIGCHERRY_RNX02_Q8_VEC", selector)
        self.assertIn("GGML_CUDA_CC_IS_RDNA3(cc)", selector)
        self.assertIn("GGML_CUDA_CC_IS_RDNA4(cc)", selector)

    def test_apply_is_idempotent_and_preserves_selector_fallback(self) -> None:
        module = _load_patch()
        source = (
            '#include "common.cuh"\n'
            '#include "fattn.cuh"\n'
            '    const bool can_use_vector_kernel = Q->ne[0] <= 256 && Q->ne[0] % 64 == 0 && Q->ne[0] != 192 && K->ne[1] % FATTN_KQ_STRIDE == 0;\n'
            '    if (can_use_vector_kernel) { return BEST_FATTN_KERNEL_VEC; }\n'
        )
        with tempfile.TemporaryDirectory(prefix="rnx02-patch-") as temp:
            root = Path(temp)
            target = root / "ggml/src/ggml-cuda/fattn.cu"
            target.parent.mkdir(parents=True)
            target.write_text(source, encoding="utf-8")
            first = apply_all(module.PATCHES, root)
            after_first = target.read_text(encoding="utf-8")
            second = apply_all(module.PATCHES, root)
            after_second = target.read_text(encoding="utf-8")

        self.assertTrue(first[0].changed)
        self.assertFalse(second[0].changed)
        self.assertEqual(after_first, after_second)
        self.assertIn("BIGCHERRY_RNX02_Q8_VEC", after_first)
        self.assertIn("if (can_use_vector_kernel)", after_first)


if __name__ == "__main__":
    unittest.main()
