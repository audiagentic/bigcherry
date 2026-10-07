"""Offline mechanics tests for 1343_mtp_nextn_rereserve."""

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
_F = "src/llama-context.cpp"


def _load():
    spec = importlib.util.spec_from_file_location("patch_1343", _REPO / "patches/1343_mtp_nextn_rereserve/patch.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_P = _load()


@unittest.skipUnless((_V / _F).exists(), "pinned vendor checkout not present")
class Patch1343Mechanics(unittest.TestCase):
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
            fn = src[src.index("void llama_context::set_embeddings_nextn(bool value, bool masked) {"):]
            fn = fn[:fn.index("void llama_context::set_embeddings_layer_inp(")]
            # the reserve is asked for only on a real change, and before the fields are assigned
            cond = "if (bigcherry_mtp_rereserve && (cparams.embeddings_nextn != value || cparams.embeddings_nextn_masked != masked)) {"
            self.assertIn(cond, fn)
            self.assertLess(fn.index(cond), fn.index("sched_need_reserve = true;"))
            self.assertLess(fn.index("sched_need_reserve = true;"), fn.index("cparams.embeddings_nextn        = value;"))
            self.assertEqual(fn.count("sched_need_reserve = true;"), 1)
            # on by default (0 disables), and the setter itself reserves or synchronizes nothing
            self.assertIn("return s == nullptr || std::atoi(s) != 0;", fn)
            self.assertNotIn("sched_reserve()", fn)
            self.assertNotIn("synchronize()", fn)
            self.assertIn("BIGCHERRY_PATCH_HIT patch=1343_mtp_nextn_rereserve", fn)
            self.assertIn("#include <cstdlib>  // bigcherry 1343", src)
            second = apply_all(_P.PATCHES, root)
            self.assertTrue(all(r.ok for r in second), [e.detail for r in second for e in r.failed])
            self.assertEqual(src, (root / _F).read_text(encoding="utf-8"))

    def test_changed_setter_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._root(td)
            p = root / _F
            before = p.read_text(encoding="utf-8").replace("    cparams.embeddings_nextn_masked = masked;\n}\n",
                                                           "    cparams.embeddings_nextn_masked = masked;\n    sched_need_reserve = true;\n}\n")
            p.write_text(before, encoding="utf-8")
            res = apply_all(_P.PATCHES, root)
            self.assertFalse(all(r.ok for r in res))
            self.assertEqual(before, p.read_text(encoding="utf-8"))

    def test_env_doc(self):
        self.assertEqual([doc.name for doc in _P.ENV_DOCS], ["BIGCHERRY_MTP_RERESERVE"])


if __name__ == "__main__":
    unittest.main()
