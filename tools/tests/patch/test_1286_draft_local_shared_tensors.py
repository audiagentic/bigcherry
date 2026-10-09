"""Offline mechanics tests for 1286_draft_local_shared_tensors (pinned llama-cparams.h, llama-context.{h,cpp},
models/dflash.cpp; alone and composed with the promoted 1303 attention split, which also edits llama-context.cpp)."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bigcherry.core import paths  # noqa: E402
from bigcherry.patcher import apply_all  # noqa: E402
from bigcherry.patch.pinned_source import copy_pinned  # noqa: E402

_REPO = Path(__file__).resolve().parents[3]
_V = paths.llama_root()
_FILES = ("src/llama-cparams.h", "src/llama-context.h", "src/llama-context.cpp", "src/models/dflash.cpp")


def _load(pid: str):
    spec = importlib.util.spec_from_file_location("patch_" + pid, _REPO / "engines" / "llamacpp" / "patches" / pid / "patch.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_P = _load("1286_draft_local_shared_tensors")


@unittest.skipUnless(all((_V / f).exists() for f in _FILES), "pinned vendor checkout not present")
class Patch1286Mechanics(unittest.TestCase):
    def _tree(self, td, extra=()):
        root = Path(td)
        for f in (*_FILES, *extra):
            (root / f).parent.mkdir(parents=True, exist_ok=True)
            copy_pinned(_V / f, root / f)
        return root

    def test_apply_and_idempotent(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._tree(td)
            res = apply_all(_P.PATCHES, root)
            self.assertTrue(all(r.ok for r in res), [e.detail for r in res for e in r.failed])
            snap = {f: (root / f).read_text(encoding="utf-8") for f in _FILES}
            self.assertIn("BigCherry 1286: draft-local copies", snap["src/llama-cparams.h"])
            self.assertIn("BigCherry 1286: a single-device DFlash/DSpark draft", snap["src/llama-context.cpp"])
            # both dflash borrow sites (tok_embd, output) prefer the draft-local copy
            dfl = snap["src/models/dflash.cpp"]
            self.assertEqual(dfl.count("tok_embd = cparams.other_tok_embd ? cparams.other_tok_embd : model_other->tok_embd;"), 2)
            self.assertEqual(dfl.count("output   = cparams.other_output ? cparams.other_output   : model_other->output;"), 2)
            self.assertEqual(dfl.count("output_s = cparams.other_output ? cparams.other_output_s : model_other->output_s;"), 2)
            self.assertNotIn("        tok_embd = model_other->tok_embd;\n", dfl)
            second = apply_all(_P.PATCHES, root)
            self.assertTrue(all(r.ok for r in second), [e.detail for r in second for e in r.failed])
            self.assertEqual(snap, {f: (root / f).read_text(encoding="utf-8") for f in _FILES})

    def test_composes_with_1303(self):
        p1303 = _load("1303_attn_kv_tensor_split")
        extra = tuple(sorted({p.path for p in p1303.PATCHES} - set(_FILES)))
        with tempfile.TemporaryDirectory() as td:
            root = self._tree(td, extra)
            res = apply_all(p1303.PATCHES, root)
            self.assertTrue(all(r.ok for r in res), [e.detail for r in res for e in r.failed])
            res = apply_all(_P.PATCHES, root)
            self.assertTrue(all(r.ok for r in res), [e.detail for r in res for e in r.failed])


if __name__ == "__main__":
    unittest.main()
