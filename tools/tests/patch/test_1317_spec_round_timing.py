"""Offline mechanics tests for 1317_spec_round_timing (pinned tools/server/server-context.cpp)."""

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
_REL = "tools/server/server-context.cpp"
_VENDOR = _REPO / "vendor/llama.cpp" / _REL
_spec = importlib.util.spec_from_file_location("patch_1317", _REPO / "patches/1317_spec_round_timing/patch.py")
assert _spec is not None and _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


@unittest.skipUnless(_VENDOR.exists(), "pinned vendor checkout not present")
class Patch1317Mechanics(unittest.TestCase):
    def test_apply_and_idempotent(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = root / _REL
            path.parent.mkdir(parents=True)
            shutil.copy2(_VENDOR, path)
            results = apply_all(_module.PATCHES, root)
            self.assertTrue(all(r.ok for r in results), [e.detail for r in results for e in r.failed])
            out = path.read_text(encoding="utf-8")
            # phase order inside one round: draft, target submit/sync, draft catch-up, sample, log
            order = [out.index(k) for k in ("bc_spec_t().draft_us +=", "bc_spec_t().sync_us   +=",
                                            "bc_spec_t().process_us +=", "const int64_t bc_ts0 = bc_spec_timing_on() ? ggml_time_us() : 0;",
                                            "BIGCHERRY_SPEC_TIMING draft_us=")]
            self.assertEqual(order, sorted(order))
            self.assertLess(out.index("struct bc_spec_timing {"), order[0])
            # the target sync stays right after llama_process, inside the same yield
            sub = out.index("ret = llama_process(ctx_tgt, LLAMA_PROCESS_TYPE_DECODE, batch.view.get());")
            self.assertLess(sub, out.index("llama_synchronize(ctx_tgt);", sub))
            self.assertLess(out.index("llama_synchronize(ctx_tgt);", sub), order[1])
            second = apply_all(_module.PATCHES, root)
            self.assertTrue(all(r.ok for r in second), [e.detail for r in second for e in r.failed])
            self.assertEqual(out, path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
