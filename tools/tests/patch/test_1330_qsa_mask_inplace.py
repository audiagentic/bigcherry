"""Offline mechanics tests for 1330_qsa_mask_inplace composed after production 1332."""

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
_HDR = "src/models/models.h"
_REL_GGML = "ggml/src/ggml.c"
_VENDOR = _REPO / "vendor/llama.cpp"


def _load(pid: str):
    spec = importlib.util.spec_from_file_location("patch_" + pid, _REPO / "patches" / pid / "patch.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_P1330 = _load("1330_qsa_mask_inplace")
_P1332 = _load("1332_qsa_token_chunk")


def _only(mod, rel):
    return [p for p in mod.PATCHES if p.path == rel]


@unittest.skipUnless((_VENDOR / _REL).exists(), "pinned vendor checkout not present")
class Patch1330Mechanics(unittest.TestCase):
    def _tree(self, td):
        root = Path(td)
        for rel in (_REL, _HDR, _REL_GGML):
            dst = root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            copy_pinned(_VENDOR / rel, dst)
        return root

    def _apply(self, root, mod):
        res = apply_all(mod.PATCHES, root)
        self.assertTrue(all(r.ok for r in res), (mod.__name__, [e.detail for r in res for e in r.failed]))

    def test_apply_after_1332_and_idempotent(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._tree(td)
            self._apply(root, _P1332)
            self._apply(root, _P1330)

            out = (root / _REL).read_text(encoding="utf-8")
            self.assertIn("bigcherry 1332: build_attn_qsa builds the masks per token chunk", out)
            self.assertIn("sel = bc_inplace ? ggml_add_inplace(ctx0, sel, kq_mask) : ggml_add(ctx0, sel, kq_mask);", out)
            self.assertIn("const int64_t bc_mask_rows = bc_mask_inplace ? GGML_PAD(n_kv + n_sel, 256) : n_kv + n_sel;", out)
            self.assertIn("bigcherry 1330: 1332's dense fallback can pass the in-place mask", out)
            attn = out[out.index("ggml_tensor * llama_model_qwen4exp::graph::build_attn_qsa("):out.index("ggml_tensor * llama_model_qwen4exp::graph::build_layer_attn(")]
            self.assertLess(attn.index("if (sel->type == GGML_TYPE_I32)"), attn.index("bigcherry 1330: 1332's dense fallback"))

            g = (root / _REL_GGML).read_text(encoding="utf-8")
            self.assertIn("bigcherry 1332: rows must be contiguous", g)
            self.assertNotIn("bigcherry 1330: rows must be contiguous", g)

            before = {rel: (root / rel).read_bytes() for rel in (_REL, _HDR, _REL_GGML)}
            self._apply(root, _P1330)
            self.assertEqual(before, {rel: (root / rel).read_bytes() for rel in (_REL, _HDR, _REL_GGML)})

    def test_composes_with_production_qwen4exp_patches(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._tree(td)
            for pid in ("1297_draft_vocab_trim", "1308_qwen4exp_rollback_copy_no_cont", "1327_qsa_host_remap"):
                res = apply_all(_only(_load(pid), _REL), root)
                self.assertTrue(all(r.ok for r in res), (pid, [e.detail for r in res for e in r.failed]))
            self._apply(root, _P1332)
            self._apply(root, _P1330)


if __name__ == "__main__":
    unittest.main()
