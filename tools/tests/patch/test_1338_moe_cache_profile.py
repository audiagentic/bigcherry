"""Offline mechanics tests for 1338_moe_cache_profile on top of b11474-rebased 1337."""

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
_VENDOR = paths.llama_root()
_CACHE = "src/llama-moe-cache.cpp"
_NEW = ("src/llama-moe-cache.cpp", "src/llama-moe-cache.h")


def _load(patch_id: str):
    spec = importlib.util.spec_from_file_location("patch_" + patch_id[:4], _REPO / "engines" / "llamacpp" / "patches" / patch_id / "patch.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_P = _load("1338_moe_cache_profile")
_P1337 = _load("1337_moe_expert_caching")
_FILES = sorted({fp.path for fp in _P1337.PATCHES} - set(_NEW))


@unittest.skipUnless(all((_VENDOR / path).exists() for path in _FILES), "pinned vendor checkout not present")
class Patch1338Mechanics(unittest.TestCase):
    def _root(self, td: str):
        root = Path(td)
        for path in _FILES:
            dst = root / path
            dst.parent.mkdir(parents=True, exist_ok=True)
            copy_pinned(_VENDOR / path, dst)
        return root

    def _apply(self, root: Path, *modules):
        for module in modules:
            res = apply_all(module.PATCHES, root)
            self.assertTrue(all(r.ok for r in res), (module.__name__, [e.detail for r in res for e in r.failed]))

    def test_apply_on_1337_and_idempotent(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._root(td)
            self._apply(root, _P1337, _P)
            src = (root / _CACHE).read_text(encoding="utf-8")
            pin = src[src.index("int32_t pin(int32_t key) {"):src.index("    std::vector<uint32_t> seen; // [n_expert]")]
            for needle in ("if (head < 0 || head == tail || slot_of[key] >= 0) {", "head = next[s];", "pinned[s] = 1;", "n_pinned++;"):
                self.assertIn(needle, pin)
            self.assertIn("if (s == tail || pinned[s]) {", src)
            self.assertIn("pinned.assign(n_slots, 0); // BigCherry 1338", src)
            plan = src[src.index("    bool plan("):src.index("// gate, up, down or gate_up, down")]
            self.assertLess(plan.index("if (uniq.size() > (size_t) n_slots) {"), plan.index("bc_unpinned > (size_t) (n_slots - n_pinned)"))
            self.assertLess(plan.index("bc_unpinned > (size_t) (n_slots - n_pinned)"), plan.index("// hits go to the tail first"))
            self.assertIn("groups[b.ig].n_slots - groups[b.ig].lru.n_pinned", src)
            self.assertIn("bc_large_batches = large != nullptr ? atoi(large) != 0 : bc_pinned > 0;", src)
            ctor = src[src.index("    impl(const llama_model & model"):src.index("    bool resolve(")]
            self.assertLess(ctor.index("buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft));"), ctor.index("bc_prewarm(n_expert); // BigCherry 1338"))
            prep = src[src.index("    bool prepare("):src.index("    void log_stats() const {")]
            self.assertLess(prep.index("if (l.planned_epoch == epoch) {"), prep.index("bc_freq[(size_t) entry->il*bc_n_expert + ids[i]]++;"))
            self.assertIn('memcmp(magic, "STRP", 4) == 0', src)
            self.assertIn("layer / expert counts do not match the model", src)
            self.assertIn("const int64_t tail = groups[ig].n_slots >= 2*n_expert ? n_expert : 8*n_expert_used;", src)
            self.assertIn("BIGCHERRY_PATCH_HIT patch=1337_moe_expert_caching", src)
            self.assertIn("BIGCHERRY_PATCH_HIT patch=1338_moe_cache_profile", src)
            before = src
            self._apply(root, _P)
            self.assertEqual(before, (root / _CACHE).read_text(encoding="utf-8"))

    def test_requires_1337(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._root(td)
            res = apply_all(_P.PATCHES, root)
            self.assertFalse(all(r.ok for r in res))


if __name__ == "__main__":
    unittest.main()
