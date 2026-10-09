"""Offline mechanics tests for 1321_mtp_ahead_primitives (pinned common/speculative.{h,cpp}; alone, with 1315 + 1318)."""

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
_V = _REPO / "vendor/llama.cpp"
_FILES = ("common/speculative.h", "common/speculative.cpp")


def _load(pid: str):
    spec = importlib.util.spec_from_file_location("patch_" + pid, _REPO / "engines" / "llamacpp" / "patches" / pid / "patch.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_P = _load("1321_mtp_ahead_primitives")


@unittest.skipUnless(all((_V / f).exists() for f in _FILES), "pinned vendor checkout not present")
class Patch1321Mechanics(unittest.TestCase):
    def _tree(self, td):
        root = Path(td)
        for f in _FILES:
            (root / f).parent.mkdir(parents=True, exist_ok=True)
            copy_pinned(_V / f, root / f)
        return root

    def test_apply_and_idempotent(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._tree(td)
            res = apply_all(_P.PATCHES, root)
            self.assertTrue(all(r.ok for r in res), [e.detail for r in res for e in r.failed])
            h = (root / "common/speculative.h").read_text(encoding="utf-8")
            self.assertLess(h.index("uint32_t seed = LLAMA_DEFAULT_SEED;"), h.index("const llama_tokens * forced = nullptr;"))
            self.assertIn("llama_tokens *       tail   = nullptr;", h)
            cpp = (root / "common/speculative.cpp").read_text(encoding="utf-8")
            mtp = cpp[cpp.index("struct common_speculative_impl_draft_mtp"):cpp.index("struct common_speculative_impl_ngram_simple")]
            # forced token chosen before the p_min gate, which skips forced tokens
            self.assertLess(mtp.index("const llama_token id = bc_forced ? (*dp.forced)[result.size()]"),
                            mtp.index("if (!bc_forced && cur_p->data[0].p < params.p_min)"))
            self.assertNotIn("tail_p_min", h + mtp)  # the tail confidence cut was removed (RV4217)
            self.assertIn("(dp.result_q == nullptr && !params.probabilistic)", mtp)
            # the front stops at n_max only without a tail; tail tokens never enter result
            self.assertIn("if (bc_front <= result.size() && dp.n_tail <= 0)", mtp)
            self.assertIn("const size_t bc_front    = dp.forced ? bc_n_forced : (size_t) params.n_max;", mtp)
            self.assertIn("dp.tail->push_back(id);", mtp)
            # the tail reports its lowest draft probability, reset when a tail draft starts
            self.assertIn("float *              tail_min_p = nullptr;", h)
            self.assertIn("*dp.tail_min_p = 1.0f;", mtp)
            self.assertIn("*dp.tail_min_p = std::min(*dp.tail_min_p, cur_p->data[0].p);", mtp)
            self.assertIn("GGML_ASSERT(dp.forced == nullptr || (int) dp.forced->size() <= params.n_max);", mtp)
            self.assertIn("bigcherry 1321: a tail only continues a front that is verified", mtp)
            snap = {f: (root / f).read_text(encoding="utf-8") for f in _FILES}
            second = apply_all(_P.PATCHES, root)
            self.assertTrue(all(r.ok for r in second))
            self.assertEqual(snap, {f: (root / f).read_text(encoding="utf-8") for f in _FILES})

    def test_composes_with_1315_and_1318(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._tree(td)
            for pid in ("1315_mtp_draft_trace", "1318_mtp_draft_timing"):
                mod = _load(pid)
                res = apply_all([p for p in mod.PATCHES if p.path in _FILES], root)
                self.assertTrue(all(r.ok for r in res), (pid, [e.detail for r in res for e in r.failed]))
            res = apply_all(_P.PATCHES, root)
            self.assertTrue(all(r.ok for r in res), [e.detail for r in res for e in r.failed])


if __name__ == "__main__":
    unittest.main()
