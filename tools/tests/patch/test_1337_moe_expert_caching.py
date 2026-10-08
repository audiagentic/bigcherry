"""Offline mechanics tests for 1337_moe_expert_caching on b11474's native copy callback."""

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
_VENDOR = _REPO / "vendor/llama.cpp"
_NEW = ("src/llama-moe-cache.cpp", "src/llama-moe-cache.h")


def _load(patch_id: str):
    spec = importlib.util.spec_from_file_location("patch_" + patch_id[:4], _REPO / "patches" / patch_id / "patch.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_P = _load("1337_moe_expert_caching")
_P1326 = _load("1326_sched_async_host_inputs")
_FILES = sorted({fp.path for module in (_P, _P1326) for fp in module.PATCHES} - set(_NEW))


@unittest.skipUnless(all((_VENDOR / path).exists() for path in _FILES), "pinned vendor checkout not present")
class Patch1337Mechanics(unittest.TestCase):
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

    def test_apply_on_b11474_and_after_production_1326(self):
        for base in ((), (_P1326,)):
            with self.subTest(base=tuple(m.__name__ for m in base)), tempfile.TemporaryDirectory() as td:
                root = self._root(td)
                self._apply(root, *base, _P)

                be = (root / "ggml/src/ggml-backend.cpp").read_text(encoding="utf-8")
                ctx = (root / "src/llama-context.cpp").read_text(encoding="utf-8")
                self.assertIn("void ggml_backend_sched_set_moe_cache(", be)
                self.assertIn("tensor_id_copy(src_id, cur_backend_id, 0) = cached_tensor;", be)
                self.assertIn("node->src[2] = cache_entry->ids_copy;", be)
                self.assertIn("The cache owns this host expert input.", be)
                self.assertIn("ggml_backend_sched_copy_input(sched, split, input);", be)
                self.assertIn("sched->callback_moe_cache_begin(sched->callback_moe_cache_user_data);", be)
                self.assertIn("ggml_backend_sched_set_copy_callback(sched.get(), sched_copy_experts, this);", ctx)
                self.assertIn("ggml_backend_sched_set_moe_cache(sched.get(), moe_cache->backend(),", ctx)
                self.assertIn("create_sched(false);", ctx)
                self.assertIn("BIGCHERRY_PATCH_HIT patch=1337_moe_expert_caching", (root / "src/llama-moe-cache.cpp").read_text(encoding="utf-8"))
                self.assertIn("static constexpr int64_t max_batch = 32;", (root / "src/llama-moe-cache.cpp").read_text(encoding="utf-8"))
                self.assertIn("class llama_moe_cache {", (root / "src/llama-moe-cache.h").read_text(encoding="utf-8"))
                self.assertIn("llama-moe-cache.cpp", (root / "src/CMakeLists.txt").read_text(encoding="utf-8"))

                before = {path: (root / path).read_bytes() for path in _FILES + list(_NEW)}
                self._apply(root, _P)
                self.assertEqual(before, {path: (root / path).read_bytes() for path in _FILES + list(_NEW)})

    def test_native_copy_callback_anchor_drift_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._root(td)
            path = root / "src/llama-context.cpp"
            text = path.read_text(encoding="utf-8")
            text = text.replace(
                "    ggml_backend_sched_set_copy_callback(sched.get(), sched_copy_experts, this);\n",
                "    ggml_backend_sched_set_copy_callback(sched.get(), sched_copy_experts, this); // drift\n",
                1,
            )
            path.write_text(text, encoding="utf-8", newline="")
            res = apply_all(_P.PATCHES, root)
            failures = [e for r in res for e in r.failed]
            self.assertTrue(any("moe-cache-llama-context-cpp-3" in e.edit_id for e in failures))


if __name__ == "__main__":
    unittest.main()
