"""Offline mechanics tests for 1332_qsa_token_chunk (pinned src/models/qwen4exp.cpp; alone and with 1297/1308/1327)."""

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
_REL = "src/models/qwen4exp.cpp"
_VENDOR = _REPO / "vendor/llama.cpp" / _REL


def _load(pid: str):
    spec = importlib.util.spec_from_file_location("patch_" + pid, _REPO / "patches" / pid / "patch.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_P = _load("1332_qsa_token_chunk")


@unittest.skipUnless(_VENDOR.exists(), "pinned vendor checkout not present")
class Patch1332Mechanics(unittest.TestCase):
    def _tree(self, td):
        root = Path(td)
        (root / _REL).parent.mkdir(parents=True)
        shutil.copy2(_VENDOR, root / _REL)
        return root

    def test_apply_and_idempotent(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._tree(td)
            res = apply_all(_P.PATCHES, root)
            self.assertTrue(all(r.ok for r in res), [e.detail for r in res for e in r.failed])
            out = (root / _REL).read_text(encoding="utf-8")
            sel = out[out.index("ggml_tensor * llama_model_qwen4exp::graph::build_qsa_sel("):out.index("// Dense GQA self-attention")]
            # the chunked path returns the remapped indices before the dense scatter
            self.assertLess(sel.index("sel_idx = ggml_cast(ctx0, idx_f, GGML_TYPE_I32);"), sel.index("return sel_idx;"))
            self.assertLess(sel.index("return sel_idx;"), sel.index("ggml_set_rows(ctx0, mask_all, zeros"))
            attn = out[out.index("ggml_tensor * llama_model_qwen4exp::graph::build_attn_qsa("):out.index("ggml_tensor * llama_model_qwen4exp::graph::build_layer_attn(")]
            self.assertIn("if (sel->type == GGML_TYPE_I32) {", attn)
            self.assertIn("ggml_tensor * mask = ggml_add(ctx0, mv, kqm);", attn)  # out of place, contiguous
            self.assertIn("cur = cur ? ggml_concat(ctx0, cur, oc, 1) : oc;", attn)
            # cache store stays before attention; v rotation undo still applies to the concatenated output
            self.assertLess(attn.index("mctx_cur->cpy_v(ctx0, v_cur, v_idxs, il)"), attn.index("if (sel->type == GGML_TYPE_I32)"))
            self.assertLess(attn.index("ggml_concat(ctx0, cur, oc, 1)"), attn.index("cb(cur, \"kqv_out\", il);"))
            second = apply_all(_P.PATCHES, root)
            self.assertTrue(all(r.ok for r in second))
            self.assertEqual(out, (root / _REL).read_text(encoding="utf-8"))

    def test_composes_with_production_qwen4exp_patches(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._tree(td)
            for pid in ("1297_draft_vocab_trim", "1308_qwen4exp_rollback_copy_no_cont", "1327_qsa_host_remap"):
                mod = _load(pid)
                res = apply_all([p for p in mod.PATCHES if p.path == _REL], root)
                self.assertTrue(all(r.ok for r in res), (pid, [e.detail for r in res for e in r.failed]))
            res = apply_all(_P.PATCHES, root)
            self.assertTrue(all(r.ok for r in res), [e.detail for r in res for e in r.failed])


if __name__ == "__main__":
    unittest.main()
