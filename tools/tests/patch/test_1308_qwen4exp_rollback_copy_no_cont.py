"""Offline mechanics tests for 1308_qwen4exp_rollback_copy_no_cont (pinned src/models/qwen4exp.cpp)."""

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
_VENDOR = _REPO / "vendor/llama.cpp/src/models/qwen4exp.cpp"
_spec = importlib.util.spec_from_file_location("patch_1308", _REPO / "patches/1308_qwen4exp_rollback_copy_no_cont/patch.py")
assert _spec is not None and _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


@unittest.skipUnless(_VENDOR.exists(), "pinned vendor checkout not present")
class Patch1308Mechanics(unittest.TestCase):
    def test_apply_and_idempotent(self):
        td = tempfile.TemporaryDirectory()
        with td:
            root = Path(td.name)
            path = root / "src/models/qwen4exp.cpp"
            path.parent.mkdir(parents=True)
            shutil.copy2(_VENDOR, path)
            results = apply_all(_module.PATCHES, root)
            self.assertTrue(all(r.ok for r in results), [e.detail for r in results for e in r.failed])
            out = path.read_text(encoding="utf-8")
            self.assertIn('std::getenv("BIGCHERRY_ROLLBACK_NO_CONT")', out)
            self.assertIn("#include <cstdlib>  // bigcherry 1308", out)
            self.assertIn("ggml_cpy(ctx0, bigcherry_rollback_no_cont ? tail : ggml_cont(ctx0, tail), dst)", out)
            self.assertNotIn("ggml_cpy(ctx0, ggml_cont(ctx0, tail), dst)", out)
            second = apply_all(_module.PATCHES, root)
            self.assertTrue(all(r.ok for r in second), [e.detail for r in second for e in r.failed])
            self.assertEqual(out, path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
