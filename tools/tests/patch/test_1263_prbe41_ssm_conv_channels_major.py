"""Mechanics tests for 1263's Meta split-state hunk (PATCH_02) against pristine b11233."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bigcherry.patcher import apply_all  # noqa: E402

_REPO = Path(__file__).resolve().parents[3]
_PATCH_FILE = _REPO / "patches/1263_prbe41_ssm_conv_channels_major/patch.py"
_PRISTINE_META = _REPO / "tools/lab/ssm-conv-split/vendor-b11233/ggml-backend-meta.cpp"
_spec = importlib.util.spec_from_file_location("patch_1263", _PATCH_FILE)
assert _spec is not None and _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

_FALLBACK = "return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};\n        }\n        // time-major"
_MARKER = "BIGCHERRY_PATCH_HIT patch=1263_prbe41 path=ssm_conv_channels_major_split_fallback"
_UNSAFE = "src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0 && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_1"


class MetaSplitFallbackTests(unittest.TestCase):
    def _apply(self, source: str):
        td = tempfile.TemporaryDirectory()
        root = Path(td.name)
        path = root / "ggml/src/ggml-backend-meta.cpp"
        path.parent.mkdir(parents=True)
        path.write_text(source, encoding="utf-8")
        return td, path, apply_all([_module.PATCH_02], root)

    def test_channels_major_is_replicated_under_tensor_split(self):
        td, path, results = self._apply(_PRISTINE_META.read_text(encoding="utf-8"))
        with td:
            self.assertTrue(all(r.ok for r in results), [e.detail for r in results for e in r.failed])
            text = path.read_text(encoding="utf-8")
            self.assertIn(_FALLBACK, text)
            self.assertEqual(text.count(_MARKER), 1)
            self.assertNotIn(_UNSAFE, text)
            before = text
            again = apply_all([_module.PATCH_02], path.parents[2])
            self.assertTrue(all(r.ok for r in again))
            self.assertEqual(before, path.read_text(encoding="utf-8"))

    def test_missing_anchor_fails_closed(self):
        broken = _PRISTINE_META.read_text(encoding="utf-8").replace(
            "        if (src_ss[0].axis == src_ss[1].axis) {\n            if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0) {\n",
            "        if (src_ss[0].axis != src_ss[1].axis) {\n            if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0) {\n",
            1,
        )
        td, _, results = self._apply(broken)
        with td:
            self.assertFalse(all(r.ok for r in results))


if __name__ == "__main__":
    unittest.main()
