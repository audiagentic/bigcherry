"""Offline mechanics tests for 1322_mtp_ahead_overlap (pinned tools/server/server-context.cpp; alone and with 1317)."""

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


def _load(pid: str):
    spec = importlib.util.spec_from_file_location("patch_" + pid, _REPO / "patches" / pid / "patch.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_P = _load("1322_mtp_ahead_overlap")


@unittest.skipUnless(_VENDOR.exists(), "pinned vendor checkout not present")
class Patch1322Mechanics(unittest.TestCase):
    def _tree(self, td):
        root = Path(td)
        (root / _REL).parent.mkdir(parents=True)
        shutil.copy2(_VENDOR, root / _REL)
        return root

    def test_apply_order_and_idempotent(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._tree(td)
            res = apply_all(_P.PATCHES, root)
            self.assertTrue(all(r.ok for r in res), [e.detail for r in res for e in r.failed])
            out = (root / _REL).read_text(encoding="utf-8")
            self.assertLess(out.index("static bool bc_mtp_ahead_on()"), out.index("struct server_slot {"))
            # ahead work runs strictly between the target submit and the target sync, and its KV is trimmed
            dec = out[out.index("bool decode(int32_t & n_batch, int32_t off)"):]
            submit = dec.index("ret = llama_process(ctx_tgt, LLAMA_PROCESS_TYPE_DECODE, batch.view.get());")
            ahead = dec.index("common_speculative_draft(spec.get());", submit)
            trim = dec.index("slot.spec_ckpt.pos_max + 1, -1)", ahead)
            sync = dec.index("llama_synchronize(ctx_tgt);", submit)
            self.assertLess(submit, ahead)
            self.assertLess(ahead, trim)
            self.assertLess(trim, sync)
            self.assertLess(sync, dec.index("common_speculative_process(spec.get(), batch.view);"))
            # promotion only on full acceptance of the same front with the predicted bonus token
            self.assertIn("slot.bc_ahead_tail.size() == (size_t) slot.get_n_draft_max() + 1", out)
            self.assertIn("accepted.size() == slot.spec_draft.size() + 1", out)
            self.assertIn("accepted.back() == slot.bc_ahead_tail[0]", out)
            self.assertIn("dp.n_tail   = (int32_t) slot.get_n_draft_max() + 1;", out)
            self.assertNotIn("PMIN", out)
            # promoted draft skips the serial draft but keeps the fresh-draft checkpoint path
            prom = out.index("bigcherry 1322: a promoted ahead tail is this round's draft")
            self.assertLess(out.index("slot.spec_ckpt.update_pos("), prom)
            self.assertLess(prom, out.index("drafting.push_back(&slot);", prom))
            second = apply_all(_P.PATCHES, root)
            self.assertTrue(all(r.ok for r in second))
            self.assertEqual(out, (root / _REL).read_text(encoding="utf-8"))

    def test_composes_with_1317(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._tree(td)
            res = apply_all(_load("1317_spec_round_timing").PATCHES, root)
            self.assertTrue(all(r.ok for r in res), [e.detail for r in res for e in r.failed])
            res = apply_all(_P.PATCHES, root)
            self.assertTrue(all(r.ok for r in res), [e.detail for r in res for e in r.failed])


if __name__ == "__main__":
    unittest.main()
