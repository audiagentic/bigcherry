"""Offline mechanics tests for 1330_qsa_mask_inplace (pinned qwen4exp.cpp + ggml.c, alone and with 1297 + 1308)."""

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
_REL_GGML = "ggml/src/ggml.c"
_VENDOR = _REPO / "vendor/llama.cpp" / _REL


def _load(pid: str):
    spec = importlib.util.spec_from_file_location("patch_" + pid, _REPO / "patches" / pid / "patch.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_P1330 = _load("1330_qsa_mask_inplace")


def _only(mod, rel):
    return [p for p in mod.PATCHES if p.path == rel]


@unittest.skipUnless(_VENDOR.exists(), "pinned vendor checkout not present")
class Patch1330Mechanics(unittest.TestCase):
    def _tree(self, td):
        root = Path(td)
        (root / _REL).parent.mkdir(parents=True)
        copy_pinned(_VENDOR, root / _REL)
        (root / _REL_GGML).parent.mkdir(parents=True)
        copy_pinned(_REPO / "vendor/llama.cpp" / _REL_GGML, root / _REL_GGML)
        return root

    def test_apply_and_idempotent(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._tree(td)
            res = apply_all(_P1330.PATCHES, root)
            self.assertTrue(all(r.ok for r in res), [e.detail for r in res for e in r.failed])
            out = (root / _REL).read_text(encoding="utf-8")
            hook = out.index("bigcherry 1330: add the causal mask in place")
            self.assertLess(out.index("ggml_set_rows(ctx0, mask_all, zeros"), hook)
            self.assertIn("sel = bc_inplace ? ggml_add_inplace(ctx0, sel, kq_mask) : ggml_add(ctx0, sel, kq_mask);", out)
            self.assertLess(hook, out.index('cb(sel, "indexer_sel", il);'))
            self.assertIn("ggml_are_same_shape(sel, kq_mask) ? (bc_mask_mode == 2 ? ggml_cont(ctx0, sel) : sel)", out)
            self.assertIn("const int64_t bc_mask_rows = bc_mask_inplace ? GGML_PAD(n_kv + n_sel, 256) : n_kv + n_sel;", out)
            g = (root / _REL_GGML).read_text(encoding="utf-8")
            self.assertIn("GGML_ASSERT(mask->nb[0] == ggml_type_size(mask->type));", g)
            self.assertIn("bigcherry 1330: rows must be contiguous", g)
            second = apply_all(_P1330.PATCHES, root)
            self.assertTrue(all(r.ok for r in second), [e.detail for r in second for e in r.failed])
            self.assertEqual(out, (root / _REL).read_text(encoding="utf-8"))

    def test_composes_with_production_qwen4exp_patches(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._tree(td)
            for pid in ("1297_draft_vocab_trim", "1308_qwen4exp_rollback_copy_no_cont", "1327_qsa_host_remap"):
                res = apply_all(_only(_load(pid), _REL), root)
                self.assertTrue(all(r.ok for r in res), (pid, [e.detail for r in res for e in r.failed]))
            res = apply_all(_P1330.PATCHES, root)
            self.assertTrue(all(r.ok for r in res), [e.detail for r in res for e in r.failed])


if __name__ == "__main__":
    unittest.main()
