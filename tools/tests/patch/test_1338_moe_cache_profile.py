"""Offline mechanics tests for 1338_moe_cache_profile (profile-pinned expert cache on top of 1337)."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bigcherry.patcher import apply_all  # noqa: E402

_REPO = Path(__file__).resolve().parents[3]
_V = _REPO / "vendor/llama.cpp"
_PIN = "d89651a7b205"
_CACHE = "src/llama-moe-cache.cpp"
_NEW = ("src/llama-moe-cache.cpp", "src/llama-moe-cache.h")


def _load(patch_id):
    spec = importlib.util.spec_from_file_location("patch_" + patch_id[:4], _REPO / "patches" / patch_id / "patch.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_P = _load("1338_moe_cache_profile")
_P1337 = _load("1337_moe_expert_caching")
_P1336 = _load("1336_sched_copy_callback")
_FILES = sorted({fp.path for module in (_P1337, _P1336) for fp in module.PATCHES} - set(_NEW))


def _pinned(path):
    # the pristine pinned file from the vendor repository, so a patched working tree does not matter
    try:
        res = subprocess.run(["git", "-C", str(_V), "show", f"{_PIN}:{path}"], capture_output=True, text=True,
                             encoding="utf-8", check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return res.stdout


_SRC = {path: _pinned(path) for path in _FILES}


@unittest.skipUnless(all(text is not None for text in _SRC.values()), "pinned vendor repository not present")
class Patch1338Mechanics(unittest.TestCase):
    def _root(self, td):
        root = Path(td)
        for path, text in _SRC.items():
            (root / path).parent.mkdir(parents=True, exist_ok=True)
            (root / path).write_text(text, encoding="utf-8", newline="\n")
        return root

    def _apply(self, root, *modules):
        for module in modules:
            res = apply_all(module.PATCHES, root)
            self.assertTrue(all(r.ok for r in res), (module.__name__, [e.detail for r in res for e in r.failed]))

    def test_apply_on_1337_and_idempotent(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._root(td)
            self._apply(root, _P1336, _P1337, _P)
            src = (root / _CACHE).read_text(encoding="utf-8")
            # a pinned slot leaves the LRU list, is never touched, and the head that gets evicted is never pinned
            pin = src[src.index("int32_t pin(int32_t key) {"):src.index("    std::vector<uint32_t> seen; // [n_expert]")]
            for needle in ("if (head < 0 || head == tail || slot_of[key] >= 0) {", "head = next[s];", "pinned[s] = 1;", "n_pinned++;"):
                self.assertIn(needle, pin)
            self.assertIn("if (s == tail || pinned[s]) {", src)
            self.assertIn("pinned.assign(n_slots, 0); // BigCherry 1338", src)
            # the pinned-aware capacity check comes after upstream's and before any slot is touched
            plan = src[src.index("    bool plan("):src.index("// gate, up, down or gate_up, down")]
            self.assertLess(plan.index("if (uniq.size() > (size_t) n_slots) {"), plan.index("bc_unpinned > (size_t) (n_slots - n_pinned)"))
            self.assertLess(plan.index("bc_unpinned > (size_t) (n_slots - n_pinned)"), plan.index("// hits go to the tail first"))
            # large batches only with a pinned set; upstream's slot-count gate is kept
            self.assertIn("if ((n_tokens > max_batch && !bc_large_batches) ||\n                std::min(n_tokens*node->src[2]->ne[0], b.src->ne[2]) > groups[b.ig].n_slots - groups[b.ig].lru.n_pinned) {", src)
            self.assertIn("bc_large_batches = large != nullptr ? atoi(large) != 0 : bc_pinned > 0;", src)
            # prewarm runs once the banks exist; counts are taken once per layer and graph, after the replay shortcut
            ctor = src[src.index("    impl(const llama_model & model"):src.index("    bool resolve(")]
            self.assertLess(ctor.index("buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft));"), ctor.index("bc_prewarm(n_expert); // BigCherry 1338"))
            prep = src[src.index("    bool prepare("):src.index("    void log_stats() const {")]
            self.assertLess(prep.index("if (l.planned_epoch == epoch) {"), prep.index("bc_freq[(size_t) entry->il*bc_n_expert + ids[i]]++;"))
            # profile reader fails on a foreign model, and the tail keeps room for one generation ubatch
            self.assertIn('memcmp(magic, "STRP", 4) == 0', src)
            self.assertIn("layer / expert counts do not match the model", src)
            self.assertIn("const int64_t tail = groups[ig].n_slots >= 2*n_expert ? n_expert : 8*n_expert_used;", src)
            self.assertIn("BIGCHERRY_PATCH_HIT patch=1338_moe_cache_profile", src)
            before = src
            self._apply(root, _P)
            self.assertEqual(before, (root / _CACHE).read_text(encoding="utf-8"))

    def test_requires_1337(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._root(td)
            self._apply(root, _P1336)
            res = apply_all(_P.PATCHES, root)
            self.assertFalse(all(r.ok for r in res))


if __name__ == "__main__":
    unittest.main()
