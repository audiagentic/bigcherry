"""Offline mechanics tests for 1333_mixed_batch_on_demand."""

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
_V = _REPO / "vendor/llama.cpp"
_F = "src/llama-graph.cpp"


def _load():
    spec = importlib.util.spec_from_file_location("patch_1333", _REPO / "patches/1333_mixed_batch_on_demand/patch.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_P = _load()


@unittest.skipUnless((_V / _F).exists(), "pinned vendor checkout not present")
class Patch1333Mechanics(unittest.TestCase):
    def _root(self, td):
        root = Path(td)
        (root / _F).parent.mkdir(parents=True, exist_ok=True)
        copy_pinned(_V / _F, root / _F)
        return root

    def test_apply_and_idempotent(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._root(td)
            res = apply_all(_P.PATCHES, root)
            self.assertTrue(all(r.ok for r in res), [e.detail for r in res for e in r.failed])
            src = (root / _F).read_text(encoding="utf-8")
            fn = src[src.index("ggml_tensor * llm_graph_context::build_inp_embd("):]
            fn = fn[:fn.index("res->t_inp_embd = cur;")]
            # the branch is conditional on the ubatch being mixed; the select still uses has_mixed for its count
            self.assertIn("cparams.ctx_type == LLAMA_CONTEXT_TYPE_DEFAULT &&\n        ubatch.is_mixed();\n    if (has_mixed) {", fn)
            self.assertIn("ggml_build_forward_select(gf, inps.data(), has_mixed ? 3 : 2, idx);", fn)
            self.assertIn("const int idx = ubatch.is_mixed() ? 2 : ubatch.token ? 0 : 1;", fn)
            self.assertEqual(fn.count("const bool has_mixed ="), 1)
            second = apply_all(_P.PATCHES, root)
            self.assertTrue(all(r.ok for r in second), [e.detail for r in second for e in r.failed])
            self.assertEqual(src, (root / _F).read_text(encoding="utf-8"))

    def test_changed_condition_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._root(td)
            p = root / _F
            p.write_text(p.read_text(encoding="utf-8").replace(
                "const bool has_mixed = llm_arch_supports_mixed_batch(arch) &&",
                "const bool has_mixed = llm_arch_allows_mixed_batch(arch) &&"), encoding="utf-8")
            res = apply_all(_P.PATCHES, root)
            self.assertFalse(all(r.ok for r in res))


if __name__ == "__main__":
    unittest.main()
