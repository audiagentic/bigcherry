"""Mechanics tests for 1255_nro06_adaptive_mtp_depth."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bigcherry.patcher import apply_all  # noqa: E402

_REPO = Path(__file__).resolve().parents[3]
_PATCH_FILE = _REPO / "engines/llamacpp/patches/1255_nro06_adaptive_mtp_depth/patch.py"
_spec = importlib.util.spec_from_file_location("patch_1255", _PATCH_FILE)
assert _spec is not None and _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

_SPEC = """#include <algorithm>

struct common_speculative_impl_draft_eagle3 : public common_speculative_impl {
    int n = 0;
};

struct common_speculative_impl_draft_mtp : public common_speculative_impl {
    std::vector<std::vector<float>> pending_h;
};
"""


class Patch1255Mechanics(unittest.TestCase):
    def _tree(self, text: str):
        td = tempfile.TemporaryDirectory()
        root = Path(td.name)
        (root / "common").mkdir(parents=True)
        (root / "common/speculative.cpp").write_text(text, encoding="utf-8")
        return td, root

    def test_apply_and_idempotent(self):
        td, root = self._tree(_SPEC)
        with td:
            first = apply_all(_module.PATCHES, root)
            self.assertTrue(all(r.ok for r in first), [e.detail for r in first for e in r.failed])
            out = (root / "common/speculative.cpp").read_text(encoding="utf-8")
            self.assertEqual(out.count("struct bigcherry_nro06_adaptive_mtp {"), 1)
            # the controller sits directly before its MTP consumer, after every other implementation
            self.assertLess(out.index("struct common_speculative_impl_draft_eagle3"),
                            out.index("struct bigcherry_nro06_adaptive_mtp {"))
            self.assertLess(out.index("BIGCHERRY_NRO06_ADAPTIVE_MTP_CONTROLLER_END"),
                            out.index("struct common_speculative_impl_draft_mtp"))
            # hysteresis policy: windowed acceptance with a dead band, cold start at depth 3
            self.assertIn("static constexpr int window_tokens = 32;", out)
            self.assertIn("static constexpr int climb_pct = 72;", out)
            self.assertIn("static constexpr int drop_pct = 60;", out)
            self.assertIn("n_cur = std::min(cap, std::max(floor, 3));", out)
            second = apply_all(_module.PATCHES, root)
            self.assertTrue(all(r.ok for r in second), [e.detail for r in second for e in r.failed])
            self.assertEqual(out, (root / "common/speculative.cpp").read_text(encoding="utf-8"))

    def test_missing_mtp_implementation_fails_closed(self):
        td, root = self._tree(_SPEC.replace("common_speculative_impl_draft_mtp", "common_speculative_impl_draft_other"))
        with td:
            results = apply_all(_module.PATCHES, root)
            self.assertFalse(all(r.ok for r in results))


if __name__ == "__main__":
    unittest.main()
