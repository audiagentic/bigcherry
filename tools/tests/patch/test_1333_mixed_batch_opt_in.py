"""Offline mechanics tests for 1333_mixed_batch_opt_in."""

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
_F = "src/llama-arch.cpp"


def _load():
    spec = importlib.util.spec_from_file_location("patch_1333", _REPO / "patches/1333_mixed_batch_opt_in/patch.py")
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
        shutil.copy2(_V / _F, root / _F)
        return root

    def test_apply_and_idempotent(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._root(td)
            res = apply_all(_P.PATCHES, root)
            self.assertTrue(all(r.ok for r in res), [e.detail for r in res for e in r.failed])
            src = (root / _F).read_text(encoding="utf-8")
            fn = src[src.index("bool llm_arch_supports_mixed_batch("):src.index("bool llm_arch_supports_sm_tensor(")]
            # the opt-in check runs before the upstream architecture switch, which is kept intact
            self.assertLess(fn.index('getenv("BIGCHERRY_MIXED_BATCH")'), fn.index("switch (arch) {"))
            self.assertLess(fn.index("if (!bc_mixed_batch) {\n        return false;"), fn.index("switch (arch) {"))
            self.assertIn("case LLM_ARCH_DFLASH:", fn)
            self.assertIn("default:\n            return true;", fn)
            self.assertEqual(src.count("#include <cstdlib>\n"), 1)
            second = apply_all(_P.PATCHES, root)
            self.assertTrue(all(r.ok for r in second), [e.detail for r in second for e in r.failed])
            self.assertEqual(src, (root / _F).read_text(encoding="utf-8"))

    def test_missing_predicate_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._root(td)
            p = root / _F
            p.write_text(p.read_text(encoding="utf-8").replace("bool llm_arch_supports_mixed_batch(",
                                                               "bool llm_arch_allows_mixed_batch("), encoding="utf-8")
            res = apply_all(_P.PATCHES, root)
            self.assertFalse(all(r.ok for r in res))


if __name__ == "__main__":
    unittest.main()
