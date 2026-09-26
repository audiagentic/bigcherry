"""Mechanics tests for 1269_prbe55_vk_smalln_dmmv."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bigcherry.patcher import apply_all  # noqa: E402

_REPO = Path(__file__).resolve().parents[3]
_PATCH_FILE = _REPO / "patches/1269_prbe55_vk_smalln_dmmv/patch.py"
_spec = importlib.util.spec_from_file_location("patch_1269", _PATCH_FILE)
assert _spec is not None and _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

_VK = """#include \"ggml-vulkan-common.h\"

static bool ggml_vk_should_use_mmvq(const vk_device& device, uint32_t m, uint32_t n, uint32_t k, ggml_type src0_type) {
    return true;
}

void f() {
    const bool f16_f32_kernel = src1->type == GGML_TYPE_F32;
    bool quantize_y = ctx->device->integer_dot_product && src1->type == GGML_TYPE_F32 && ggml_is_contiguous(src1) && !y_non_contig && (ne11 * ne10) % 4 == 0 && ggml_vk_should_use_mmvq(ctx->device, ne01, ne11, ne10, src0->type);

    vk_pipeline to_fp16_vk_0 = nullptr;
}
"""
_TESTS = """        test_cases.emplace_back(new test_l2_norm_batch(GGML_TYPE_F32, { n, 16, 16, 1 }, 4, 1e-12f, true));
    }


    return test_cases;
}"""


class Patch1269Mechanics(unittest.TestCase):
    def _tree(self):
        td = tempfile.TemporaryDirectory()
        root = Path(td.name)
        (root / "ggml/src/ggml-vulkan").mkdir(parents=True)
        (root / "tests").mkdir(parents=True)
        (root / "ggml/src/ggml-vulkan/ggml-vulkan.cpp").write_text(_VK, encoding="utf-8")
        (root / "tests/test-backend-ops.cpp").write_text(_TESTS, encoding="utf-8")
        return td, root

    def test_apply_and_idempotent(self):
        td, root = self._tree()
        with td:
            first = apply_all(_module.PATCHES, root)
            self.assertTrue(all(r.ok for r in first), [e.detail for r in first for e in r.failed])
            text = (root / "ggml/src/ggml-vulkan/ggml-vulkan.cpp").read_text()
            self.assertIn("n >= 2 && n <= 8", text)
            self.assertIn("BIGCHERRY_VK_SMALLN_DMMV", text)
            self.assertIn("quantize_y = false", text)
            cases = (root / "tests/test-backend-ops.cpp").read_text()
            self.assertIn("4096, 6,   4096", cases)
            self.assertIn("4096, 8,   4096", cases)
            before = {p: (root / p).read_text() for p in ("ggml/src/ggml-vulkan/ggml-vulkan.cpp", "tests/test-backend-ops.cpp")}
            second = apply_all(_module.PATCHES, root)
            self.assertTrue(all(r.ok for r in second))
            self.assertEqual(before, {p: (root / p).read_text() for p in before})

    def test_missing_selector_fails_closed(self):
        td, root = self._tree()
        with td:
            path = root / "ggml/src/ggml-vulkan/ggml-vulkan.cpp"
            path.write_text(_VK.replace("ggml_vk_should_use_mmvq(ctx->device, ne01, ne11, ne10, src0->type)", "true"), encoding="utf-8")
            results = apply_all(_module.PATCHES, root)
            self.assertFalse(all(r.ok for r in results))


if __name__ == "__main__":
    unittest.main()
