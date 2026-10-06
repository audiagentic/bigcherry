"""Offline mechanics tests for 1315_mtp_draft_trace (pinned common/speculative.cpp)."""

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
_VENDOR = _REPO / "vendor/llama.cpp/common/speculative.cpp"
_spec = importlib.util.spec_from_file_location("patch_1315", _REPO / "patches/1315_mtp_draft_trace/patch.py")
assert _spec is not None and _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


@unittest.skipUnless(_VENDOR.exists(), "pinned vendor checkout not present")
class Patch1315Mechanics(unittest.TestCase):
    def test_apply_and_idempotent(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = root / "common/speculative.cpp"
            path.parent.mkdir(parents=True)
            copy_pinned(_VENDOR, path)
            results = apply_all(_module.PATCHES, root)
            self.assertTrue(all(r.ok for r in results), [e.detail for r in results for e in r.failed])
            out = path.read_text(encoding="utf-8")
            self.assertLess(out.index("static uint64_t bc_fnv1a("), out.index("BIGCHERRY_DRAFT_TRACE step"))
            step = out.index("BIGCHERRY_DRAFT_TRACE step")
            self.assertLess(out.index("const float * h_row = llama_get_embeddings_nextn_ith(ctx_dft, i_last[seq_id]);"), step)
            accept = out.index("BIGCHERRY_DRAFT_TRACE accept")
            self.assertLess(out.index("verify_h[seq_id].data() + (size_t) i_h * n_embd, row_bytes);"), accept)
            second = apply_all(_module.PATCHES, root)
            self.assertTrue(all(r.ok for r in second), [e.detail for r in second for e in r.failed])
            self.assertEqual(out, path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
