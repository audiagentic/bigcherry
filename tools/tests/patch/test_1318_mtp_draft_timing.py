"""Offline mechanics tests for 1318_mtp_draft_timing (pinned common/speculative.cpp, alone and with 1315)."""

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


def _load(pid: str):
    spec = importlib.util.spec_from_file_location("patch_" + pid, _REPO / "patches" / pid / "patch.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_P1315 = _load("1315_mtp_draft_trace")
_P1318 = _load("1318_mtp_draft_timing")


@unittest.skipUnless(_VENDOR.exists(), "pinned vendor checkout not present")
class Patch1318Mechanics(unittest.TestCase):
    def _apply(self, mods):
        td = tempfile.TemporaryDirectory()
        root = Path(td.name)
        path = root / "common/speculative.cpp"
        path.parent.mkdir(parents=True)
        copy_pinned(_VENDOR, path)
        for mod in mods:
            res = apply_all(mod.PATCHES, root)
            self.assertTrue(all(r.ok for r in res), [e.detail for r in res for e in r.failed])
        return td, root, path

    def test_alone_and_idempotent(self):
        td, root, path = self._apply([_P1318])
        with td:
            out = path.read_text(encoding="utf-8")
            start = out.index("bigcherry 1318: draft-loop timing")
            submit = out.index("bc_dt_submit += bc_dt_s1 - bc_dt_s0;")
            end = out.index("BIGCHERRY_DRAFT_TIMING steps=")
            self.assertLess(start, submit)
            self.assertLess(submit, end)
            self.assertLess(end, out.index("llama_set_nextn_layer_offset(ctx_dft, 0); // restore default for non-draft decodes", end))
            second = apply_all(_P1318.PATCHES, root)
            self.assertTrue(all(r.ok for r in second), [e.detail for r in second for e in r.failed])
            self.assertEqual(out, path.read_text(encoding="utf-8"))

    def test_composes_with_1315(self):
        td, _root, path = self._apply([_P1315, _P1318])
        with td:
            out = path.read_text(encoding="utf-8")
            self.assertIn("BIGCHERRY_DRAFT_TRACE step", out)
            self.assertIn("BIGCHERRY_DRAFT_TIMING steps=", out)


if __name__ == "__main__":
    unittest.main()
