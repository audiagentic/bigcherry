"""Offline mechanics tests for 1295_qsa_gather_decode (pinned qwen4exp.cpp + models.h; alone and before 1332)."""

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
_V = _REPO / "vendor/llama.cpp"


def _load(pid: str):
    spec = importlib.util.spec_from_file_location("patch_" + pid, _REPO / "patches" / pid / "patch.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_P = _load("1295_qsa_gather_decode")
_FILES = tuple(sorted({p.path for p in _P.PATCHES}))


@unittest.skipUnless(all((_V / f).exists() for f in _FILES), "pinned vendor checkout not present")
class Patch1295Mechanics(unittest.TestCase):
    def _tree(self, td, extra=()):
        root = Path(td)
        for f in (*_FILES, *extra):
            (root / f).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(_V / f, root / f)
        return root

    def test_apply_and_idempotent(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._tree(td)
            res = apply_all(_P.PATCHES, root)
            self.assertTrue(all(r.ok for r in res), [e.detail for r in res for e in r.failed])
            snap = {f: (root / f).read_text(encoding="utf-8") for f in _FILES}
            cpp = snap["src/models/qwen4exp.cpp"]
            self.assertEqual(cpp.count("BigCherry 1295: small batches over a large cache attend"), 1)
            # off by default: enabled only when the variable is set and not "0"
            self.assertIn('const bool enabled = e != nullptr && strcmp(e, "0") != 0;', cpp)
            # the gather branch sits before the dense mask path of build_attn_qsa
            attn = cpp[cpp.index("ggml_tensor * llama_model_qwen4exp::graph::build_attn_qsa("):]
            self.assertLess(attn.index("kqv_out_qsa_gather"), attn.index("// the selection mask already carries the causal mask"))
            second = apply_all(_P.PATCHES, root)
            self.assertTrue(all(r.ok for r in second), [e.detail for r in second for e in r.failed])
            self.assertEqual(snap, {f: (root / f).read_text(encoding="utf-8") for f in _FILES})  # guard regression

    def test_composes_before_1332(self):
        p1332 = _load("1332_qsa_token_chunk")
        extra = tuple(sorted({p.path for p in p1332.PATCHES} - set(_FILES)))
        with tempfile.TemporaryDirectory() as td:
            root = self._tree(td, extra)
            for mod in (_P, p1332):
                res = apply_all(mod.PATCHES, root)
                self.assertTrue(all(r.ok for r in res), [e.detail for r in res for e in r.failed])
            cpp = (root / "src/models/qwen4exp.cpp").read_text(encoding="utf-8")
            attn = cpp[cpp.index("ggml_tensor * llama_model_qwen4exp::graph::build_attn_qsa("):]
            # small batches return from the gather branch before the chunked path
            self.assertLess(attn.index("kqv_out_qsa_gather"), attn.index("if (sel->type == GGML_TYPE_I32) {"))


if __name__ == "__main__":
    unittest.main()
