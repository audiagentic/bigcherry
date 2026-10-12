"""Offline mechanics tests for 1358_meta_split_cache_local_evict."""

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
_F = "ggml/src/ggml-backend-meta.cpp"


def _load():
    spec = importlib.util.spec_from_file_location(
        "patch_1358", _REPO / "engines/llamacpp/patches/1358_meta_split_cache_local_evict/patch.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_P = _load()


@unittest.skipUnless((_V / _F).exists(), "pinned vendor checkout not present")
class Patch1358Mechanics(unittest.TestCase):
    def _root(self, td):
        root = Path(td)
        (root / _F).parent.mkdir(parents=True, exist_ok=True)
        copy_pinned(_V / _F, root / _F)
        return root

    def test_apply_evicts_one_entry_and_keeps_the_off_switch(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._root(td)
            res = apply_all(_P.PATCHES, root)
            self.assertTrue(all(r.ok for r in res), [e.detail for r in res for e in r.failed])
            src = (root / _F).read_text(encoding="utf-8")
            block = src[src.index("const std::pair key = std::make_pair(tensor, assume_sync);"):]
            block = block[:block.index("if (it == buf_ctx->split_state_cache.end()) {")]
            # on: erase the stale entry; off: upstream's whole-cache clear; either way the lookup then misses
            on = block.index("if (bc_local_evict) {")
            erase = block.index("buf_ctx->split_state_cache.erase(it);")
            clear = block.index("buf_ctx->split_state_cache.clear();")
            self.assertLess(on, erase)
            self.assertLess(erase, block.index("} else {"))
            self.assertLess(block.index("} else {"), clear)
            self.assertLess(clear, block.index("it = buf_ctx->split_state_cache.end();"))
            self.assertEqual(block.count("buf_ctx->split_state_cache.erase(it);"), 1)
            self.assertIn("return s == nullptr || atoi(s) != 0;", block)
            self.assertIn('GGML_LOG_WARN("BIGCHERRY_PATCH_HIT patch=1358_meta_split_cache_local_evict', block)

            second = apply_all(_P.PATCHES, root)
            self.assertTrue(all(r.ok for r in second), [e.detail for r in second for e in r.failed])
            self.assertEqual(src, (root / _F).read_text(encoding="utf-8"))

    def test_changed_cache_reaction_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._root(td)
            p = root / _F
            before = p.read_text(encoding="utf-8").replace("        buf_ctx->split_state_cache.clear();\n        it = buf_ctx->split_state_cache.end();",
                                                           "        buf_ctx->split_state_cache.clear();\n        it = buf_ctx->split_state_cache.find(key);")
            p.write_text(before, encoding="utf-8")
            res = apply_all(_P.PATCHES, root)
            self.assertFalse(all(r.ok for r in res))
            self.assertEqual(before, p.read_text(encoding="utf-8"))

    def test_env_doc(self):
        self.assertEqual([doc.name for doc in _P.ENV_DOCS], ["BIGCHERRY_META_SPLIT_CACHE_EVICT"])


if __name__ == "__main__":
    unittest.main()
