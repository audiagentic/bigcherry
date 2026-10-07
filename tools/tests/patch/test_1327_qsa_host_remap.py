"""Offline mechanics tests for 1327_qsa_host_remap (pinned src/models/qwen4exp.cpp, alone and with 1297 + 1308)."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bigcherry.patcher import apply_all  # noqa: E402
from bigcherry.patch.pinned_source import copy_pinned  # noqa: E402

_REPO = Path(__file__).resolve().parents[3]
_REL = "src/models/qwen4exp.cpp"
_VENDOR = _REPO / "vendor/llama.cpp" / _REL


def _load(pid: str):
    spec = importlib.util.spec_from_file_location("patch_" + pid, _REPO / "patches" / pid / "patch.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_P1327 = _load("1327_qsa_host_remap")


def _only(mod, rel):
    return [p for p in mod.PATCHES if p.path == rel]


@unittest.skipUnless(_VENDOR.exists(), "pinned vendor checkout not present")
class Patch1327Mechanics(unittest.TestCase):
    def _tree(self, td):
        root = Path(td)
        (root / _REL).parent.mkdir(parents=True)
        copy_pinned(_VENDOR, root / _REL)
        return root

    def test_apply_and_idempotent(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._tree(td)
            res = apply_all(_P1327.PATCHES, root)
            self.assertTrue(all(r.ok for r in res), [e.detail for r in res for e in r.failed])
            out = (root / _REL).read_text(encoding="utf-8")
            build = out.index("bigcherry 1327: host-side QSA remap inputs")
            self.assertLess(out.index("inp->n_sel = kpool*std::min<uint32_t>"), build)
            use = out.index("bigcherry 1327: host-computed when enabled")
            self.assertLess(use, out.index("idx_f   = ggml_add(ctx0, ggml_mul(ctx0, ggml_sub(ctx0, idx_f, dump), live), dump);"))
            self.assertIn("lt[i] = t[i] < (int32_t) n_kv ? 1.0f : 0.0f;", out)
            self.assertIn("d[j*bc_dump->ne[0] + s] = (float) ((int64_t) n_kv + s);", out)
            second = apply_all(_P1327.PATCHES, root)
            self.assertTrue(all(r.ok for r in second), [e.detail for r in second for e in r.failed])
            self.assertEqual(out, (root / _REL).read_text(encoding="utf-8"))

    def test_composes_with_production_qwen4exp_patches(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._tree(td)
            for pid in ("1297_draft_vocab_trim", "1308_qwen4exp_rollback_copy_no_cont"):
                res = apply_all(_only(_load(pid), _REL), root)
                self.assertTrue(all(r.ok for r in res), (pid, [e.detail for r in res for e in r.failed]))
            res = apply_all(_P1327.PATCHES, root)
            self.assertTrue(all(r.ok for r in res), [e.detail for r in res for e in r.failed])


if __name__ == "__main__":
    unittest.main()
