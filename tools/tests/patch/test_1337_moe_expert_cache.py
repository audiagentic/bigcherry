"""Offline mechanics tests for 1337_moe_expert_cache (backport of upstream #29887, on top of 1336)."""

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
_BE = "ggml/src/ggml-backend.cpp"
_CTX = "src/llama-context.cpp"
_NEW = ("src/llama-moe-cache.cpp", "src/llama-moe-cache.h")


def _load(patch_id):
    spec = importlib.util.spec_from_file_location("patch_" + patch_id[:4], _REPO / "patches" / patch_id / "patch.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_P = _load("1337_moe_expert_cache")
_P1336 = _load("1336_sched_copy_callback")
_P1326 = _load("1326_sched_async_host_inputs")
_FILES = sorted({fp.path for module in (_P, _P1336, _P1326) for fp in module.PATCHES} - set(_NEW))


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
class Patch1337Mechanics(unittest.TestCase):
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

    def _read(self, root):
        return {path: (root / path).read_text(encoding="utf-8") for path in list(_FILES) + list(_NEW)}

    def test_apply_on_1336_with_and_without_1326_and_idempotent(self):
        for base in ((_P1336,), (_P1326, _P1336)):
            with tempfile.TemporaryDirectory() as td:
                root = self._root(td)
                self._apply(root, *base, _P)
                out = self._read(root)
                be, ctx = out[_BE], out[_CTX]
                # the two new files exist and the library builds them
                self.assertIn("class llama_moe_cache {", out[_NEW[1]])
                self.assertIn("llama-moe-cache.cpp", (root / "src/CMakeLists.txt").read_text(encoding="utf-8"))
                # scheduler hooks: setter, split-graph substitution, compute-time upload
                self.assertEqual(be.count("void ggml_backend_sched_set_moe_cache("), 1)
                self.assertIn("tensor_id_copy(src_id, cur_backend_id, 0) = cached_tensor;", be)
                self.assertIn("node->src[2] = cache_entry->ids_copy;", be)
                self.assertIn("} else if (cache_entry != NULL) {", be)
                self.assertIn("sched->callback_moe_cache_begin(sched->callback_moe_cache_user_data);", be)
                # 1336's copy callback is still the path for uncached host weights, and both live in one loop
                self.assertIn("sched->callback_copy(split_backend, input, input_cpy, &split->graph, sched->callback_copy_user_data)", be)
                self.assertLess(be.index("bc_copy_pass < 2"), be.index("ggml_backend_sched_moe_cache_entry_find(sched, split_id, input, NULL);"))
                # upstream's create_sched installs both the copy callback and the cache
                lam = ctx[ctx.index("auto create_sched = [&](bool parallel) {"):ctx.index("create_sched(cparams.pipeline_parallel);")]
                self.assertIn("ggml_backend_sched_set_copy_callback(sched.get(), sched_copy_experts, this);", lam)
                self.assertIn("ggml_backend_sched_set_moe_cache(sched.get(), moe_cache->backend(),", lam)
                self.assertEqual(ctx.count("ggml_backend_sched_set_copy_callback(sched.get(), sched_copy_experts, this);"), 1)
                self.assertIn("create_sched(false);", ctx)
                # the flag and its plumbing
                self.assertIn('{"--moe-cache-mib"}, "N",', (root / "common/arg.cpp").read_text(encoding="utf-8"))
                self.assertIn("cparams.moe_cache_size = params.moe_cache_size;", (root / "common/common.cpp").read_text(encoding="utf-8"))
                self._apply(root, _P)
                self.assertEqual(out, self._read(root))

    def test_requires_1336(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._root(td)
            res = apply_all(_P.PATCHES, root)
            self.assertFalse(all(r.ok for r in res))


if __name__ == "__main__":
    unittest.main()
