"""Offline mechanics tests for 1311_hc_pre_q81 (pinned ggml/src/ggml-cuda/dsv4-hc.cu)."""

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
_VENDOR = _REPO / "vendor/llama.cpp/ggml/src/ggml-cuda/dsv4-hc.cu"
_spec = importlib.util.spec_from_file_location("patch_1311", _REPO / "patches/1311_hc_pre_q81/patch.py")
assert _spec is not None and _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


@unittest.skipUnless(_VENDOR.exists(), "pinned vendor checkout not present")
class Patch1311Mechanics(unittest.TestCase):
    def test_apply_and_idempotent(self):
        td = tempfile.TemporaryDirectory()
        with td:
            root = Path(td.name)
            path = root / "ggml/src/ggml-cuda/dsv4-hc.cu"
            path.parent.mkdir(parents=True)
            shutil.copy2(_VENDOR, path)
            results = apply_all(_module.PATCHES, root)
            self.assertTrue(all(r.ok for r in results), [e.detail for r in results for e in r.failed])
            out = path.read_text(encoding="utf-8")
            self.assertIn("static __global__ void dsv4_hc_pre_q81_f32(", out)
            self.assertIn("q81, dst, dst->data, ctx.curr_stream_no, n_embd, n_embd_padded, n_tokens,", out)
            self.assertIn('tokens=%lld\\n"', out)
            gate = out.index("bigcherry 1311: emit the Q8_1 activation")
            self.assertLess(gate, out.index("    auto kernel = gated ? dsv4_hc_pre_f32<true> : dsv4_hc_pre_f32<false>;"))
            self.assertLess(out.index("static __global__ void dsv4_hc_pre_q81_f32("), gate)
            second = apply_all(_module.PATCHES, root)
            self.assertTrue(all(r.ok for r in second), [e.detail for r in second for e in r.failed])
            self.assertEqual(out, path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
