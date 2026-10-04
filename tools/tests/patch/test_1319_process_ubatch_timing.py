"""Offline mechanics tests for 1319_process_ubatch_timing (pinned src/llama-context.cpp)."""

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
_VENDOR = _REPO / "vendor/llama.cpp/src/llama-context.cpp"
_spec = importlib.util.spec_from_file_location("patch_1319", _REPO / "patches/1319_process_ubatch_timing/patch.py")
assert _spec is not None and _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


@unittest.skipUnless(_VENDOR.exists(), "pinned vendor checkout not present")
class Patch1319Mechanics(unittest.TestCase):
    def test_apply_and_idempotent(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = root / "src/llama-context.cpp"
            path.parent.mkdir(parents=True)
            shutil.copy2(_VENDOR, path)
            results = apply_all(_module.PATCHES, root)
            self.assertTrue(all(r.ok for r in results), [e.detail for r in results for e in r.failed])
            out = path.read_text(encoding="utf-8")
            start = out.index("bigcherry 1319: host-time split of process_ubatch")
            mid = out.index("bigcherry 1319: graph build or reuse done")
            inputs = out.index("bigcherry 1319: inputs set")
            log = out.index("BIGCHERRY_SUBMIT_TIMING ctx=")
            self.assertLess(start, mid)
            self.assertLess(mid, out.index("res->set_inputs(&ubatch);", mid))
            self.assertLess(inputs, out.index("const auto status = graph_compute(res->get_gf(), ubatch.n_tokens > 1);"))
            self.assertLess(out.index("const auto status = graph_compute(res->get_gf(), ubatch.n_tokens > 1);"), log)
            self.assertIn("#include <cstdlib>  // bigcherry 1319", out)
            second = apply_all(_module.PATCHES, root)
            self.assertTrue(all(r.ok for r in second), [e.detail for r in second for e in r.failed])
            self.assertEqual(out, path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
