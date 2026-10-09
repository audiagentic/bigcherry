"""Mechanics tests for 1263: channels-major SSM_CONV, declined under a Meta tensor split."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bigcherry.patcher import apply_all  # noqa: E402

_REPO = Path(__file__).resolve().parents[3]
_VENDOR = _REPO / "vendor/llama.cpp"
_spec = importlib.util.spec_from_file_location(
    "patch_1263", _REPO / "engines/llamacpp/patches/1263_prbe41_ssm_conv_channels_major/patch.py")
assert _spec is not None and _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

_MARKER = "BIGCHERRY_PATCH_HIT patch=1263_prbe41 path=ssm_conv_channels_major_declined_split"


def _pristine(path: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(_VENDOR), "show", f"HEAD:{path}"])


@unittest.skipUnless((_VENDOR / ".git").exists(), "pinned vendor checkout required")
class Patch1263Mechanics(unittest.TestCase):
    def _apply(self):
        td = tempfile.TemporaryDirectory()
        root = Path(td.name)
        for file_patch in _module.PATCHES:
            dst = root / file_patch.path
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(_pristine(file_patch.path))
        return td, root, apply_all(_module.PATCHES, root)

    def test_meta_backend_is_left_pristine(self):
        self.assertNotIn("ggml/src/ggml-backend-meta.cpp", {p.path for p in _module.PATCHES})

    def test_tensor_split_takes_the_pristine_time_major_graph(self):
        td, root, results = self._apply()
        with td:
            self.assertTrue(all(r.ok for r in results), [e.detail for r in results for e in r.failed])
            base = (root / "src/models/delta-net-base.cpp").read_text(encoding="utf-8")
            self.assertEqual(base.count(_MARKER), 1)
            self.assertIn(_MARKER + '\\n");', base)  # C escape, not a raw newline
            fallback = base.split("if (!channels_major) {", 1)[1].split("return conv_input;", 1)[0]
            self.assertIn("qkv_mixed = ggml_transpose(ctx0, qkv_mixed);", fallback)
            self.assertIn("ggml_concat(ctx0, conv_states, qkv_mixed, 0);", fallback)
            # The early return precedes the channels-major body.
            self.assertLess(base.index("if (!channels_major)"), base.index("channels-major layout: [conv_channels"))
            self.assertIn("bool                 channels_major,",
                          (root / "src/models/models.h").read_text(encoding="utf-8"))
            for model in ("qwen35", "qwen35moe", "qwen3next"):
                text = (root / f"src/models/{model}.cpp").read_text(encoding="utf-8")
                self.assertIn("model.split_mode() != LLAMA_SPLIT_MODE_TENSOR", text)
                self.assertIn("conv_channels, channels_major, il)", text)
                self.assertIn("? ggml_ssm_conv_channels_major", text)
                self.assertIn(": ggml_ssm_conv(ctx0, conv_input, conv_kernel)", text)

    def test_idempotent(self):
        td, root, results = self._apply()
        with td:
            self.assertTrue(all(r.ok for r in results))
            before = {p.path: (root / p.path).read_bytes() for p in _module.PATCHES}
            again = apply_all(_module.PATCHES, root)
            self.assertTrue(all(r.ok for r in again))
            self.assertEqual(before, {p.path: (root / p.path).read_bytes() for p in _module.PATCHES})

    def test_missing_anchor_fails_closed(self):
        td = tempfile.TemporaryDirectory()
        with td:
            root = Path(td.name)
            for file_patch in _module.PATCHES:
                dst = root / file_patch.path
                dst.parent.mkdir(parents=True, exist_ok=True)
                data = _pristine(file_patch.path)
                if file_patch.path == "src/models/qwen35.cpp":
                    data = data.replace(b"conv_kernel_size, conv_channels, il);", b"conv_kernel_size, il);", 1)
                dst.write_bytes(data)
            results = apply_all(_module.PATCHES, root)
            self.assertFalse(all(r.ok for r in results))


if __name__ == "__main__":
    unittest.main()
