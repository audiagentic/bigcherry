"""Offline mechanics tests for 1325_sched_split_timing (pinned ggml/src/ggml-backend.cpp)."""

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
_VENDOR = _REPO / "vendor/llama.cpp/ggml/src/ggml-backend.cpp"
_spec = importlib.util.spec_from_file_location("patch_1325", _REPO / "patches/1325_sched_split_timing/patch.py")
assert _spec is not None and _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


@unittest.skipUnless(_VENDOR.exists(), "pinned vendor checkout not present")
class Patch1325Mechanics(unittest.TestCase):
    def test_apply_and_idempotent(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = root / "ggml/src/ggml-backend.cpp"
            path.parent.mkdir(parents=True)
            shutil.copy2(_VENDOR, path)
            results = apply_all(_module.PATCHES, root)
            self.assertTrue(all(r.ok for r in results), [e.detail for r in results for e in r.failed])
            out = path.read_text(encoding="utf-8")
            start = out.index("bigcherry 1325: per-split host timing")
            t0 = out.index("const int64_t bc_ss_t0 = bc_ss_on ? ggml_time_us() : 0;")
            log = out.index("BIGCHERRY_SCHED_SPLIT i=")
            self.assertLess(start, t0)
            self.assertLess(t0, out.index("ec = ggml_backend_graph_compute_async(split_backend, &split->graph);", t0))
            self.assertLess(out.index("ec = ggml_backend_graph_compute_async(split_backend, &split->graph);", t0), log)
            second = apply_all(_module.PATCHES, root)
            self.assertTrue(all(r.ok for r in second), [e.detail for r in second for e in r.failed])
            self.assertEqual(out, path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
