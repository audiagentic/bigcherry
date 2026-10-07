"""Offline mechanics tests for 1320_meta_compute_timing (pinned ggml/src/ggml-backend-meta.cpp)."""

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
_VENDOR = _REPO / "vendor/llama.cpp/ggml/src/ggml-backend-meta.cpp"
_spec = importlib.util.spec_from_file_location("patch_1320", _REPO / "patches/1320_meta_compute_timing/patch.py")
assert _spec is not None and _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


def _load_patch(patch_id: str):
    path = _REPO / f"patches/{patch_id}/patch.py"
    spec = importlib.util.spec_from_file_location(f"patch_{patch_id}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _only(module, path: str):
    return [patch for patch in module.PATCHES if patch.path == path]


_P1339 = _load_patch("1339_meta_memory_report")
_P1340 = _load_patch("1340_meta_per_device_arena")


@unittest.skipUnless(_VENDOR.exists(), "pinned vendor checkout not present")
class Patch1320Mechanics(unittest.TestCase):
    def test_apply_and_idempotent(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = root / "ggml/src/ggml-backend-meta.cpp"
            path.parent.mkdir(parents=True)
            copy_pinned(_VENDOR, path)
            results = apply_all(_module.PATCHES, root)
            self.assertTrue(all(r.ok for r in results), [e.detail for r in results for e in r.failed])
            out = path.read_text(encoding="utf-8")
            start = out.index("bigcherry 1320: meta graph_compute host-time breakdown")
            exe = out.index("bigcherry 1320: rebuild done, execution starts")
            launch = out.index("bc_mt_launch += bc_mt_l1 - bc_mt_l0;")
            log = out.index("BIGCHERRY_META_TIMING n_nodes=")
            self.assertLess(start, exe)
            self.assertLess(exe, launch)
            self.assertLess(launch, out.index("backend_allreduce_success = backend_ctx->comm_allreduce(", launch))
            self.assertLess(out.index("const ggml_status status = allreduce_fallback(i);", launch), log)
            self.assertIn("#include <cstdlib>  // bigcherry 1320", out)
            second = apply_all(_module.PATCHES, root)
            self.assertTrue(all(r.ok for r in second), [e.detail for r in second for e in r.failed])
            self.assertEqual(out, path.read_text(encoding="utf-8"))

    def test_composes_after_production_meta_arena(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = root / "ggml/src/ggml-backend-meta.cpp"
            path.parent.mkdir(parents=True)
            copy_pinned(_VENDOR, path)
            for patches in (
                _only(_P1339, "ggml/src/ggml-backend-meta.cpp"),
                _only(_P1340, "ggml/src/ggml-backend-meta.cpp"),
                _module.PATCHES,
            ):
                results = apply_all(patches, root)
                self.assertTrue(all(r.ok for r in results), [e.detail for r in results for e in r.failed])
            out = path.read_text(encoding="utf-8")
            self.assertIn("BIGCHERRY_META_MEM compute dev=", out)
            self.assertIn("ggml_backend_meta_per_device_arena_enabled()", out)
            self.assertIn("BIGCHERRY_META_TIMING n_nodes=", out)


if __name__ == "__main__":
    unittest.main()
