"""Offline mechanics tests for 1306_qwen4exp_shexp_split (pinned src/llama-model.cpp + 1303)."""

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
_VENDOR = _REPO / "vendor/llama.cpp/src/llama-model.cpp"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_module = _load("patch_1306", _REPO / "patches/1306_qwen4exp_shexp_split/patch.py")
_p1303 = _load("patch_1303", _REPO / "patches/1303_attn_kv_tensor_split/patch.py")


@unittest.skipUnless(_VENDOR.exists(), "pinned vendor checkout not present")
class Patch1306Mechanics(unittest.TestCase):
    def test_apply_before_output_rules_and_idempotent(self):
        td = tempfile.TemporaryDirectory()
        with td:
            root = Path(td.name)
            path = root / "src/llama-model.cpp"
            path.parent.mkdir(parents=True)
            shutil.copy2(_VENDOR, path)
            prereq = apply_all(_p1303.PATCHES, root)
            self.assertTrue(all(r.ok for r in prereq), [e.detail for r in prereq for e in r.failed])
            results = apply_all(_module.PATCHES, root)
            self.assertTrue(all(r.ok for r in results), [e.detail for r in results for e in r.failed])
            out = path.read_text(encoding="utf-8")
            self.assertIn("ud->model->arch == LLM_ARCH_QWEN4EXP", out)
            self.assertIn('std::getenv("BIGCHERRY_SHEXP_SPLIT")', out)
            block = out[out.index("bigcherry 1306"):out.index("        // output\n")]
            self.assertIn('GGML_BACKEND_SPLIT_AXIS_1, "ffn_down_shexp.weight"', block)
            self.assertIn('GGML_BACKEND_SPLIT_AXIS_0, "ffn_down_shexp.weight"', block)
            self.assertIn('1306_shexp_split\\n"', block)
            # 1303 now provides <atomic> for the once-only log flags.
            self.assertIn("#include <atomic>   // bigcherry 1303", out)
            # The rule must precede the final MIRRORED fallthrough of get_tensor_config.
            self.assertLess(out.index("bigcherry 1306"), out.index("        // everything else\n        return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_MIRRORED);"))

            second = apply_all(_module.PATCHES, root)
            self.assertTrue(all(r.ok for r in second), [e.detail for r in second for e in r.failed])
            self.assertEqual(out, path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
