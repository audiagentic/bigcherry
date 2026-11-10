"""VTP11 opt-in patch safety mechanics (no GPU, no pinned source checkout required)."""

from __future__ import annotations

import importlib.util
import re
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools"))
from bigcherry.patcher import apply_all

PACKAGE = (
    REPO / "engines" / "radiance" / "patches" /
    "rad0014_vtp11_host_device_guard" / "patch.py"
)


def patches():
    spec = importlib.util.spec_from_file_location("rad0014_vtp11", PACKAGE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return list(mod.PATCHES)


def _unescape_literal(pattern: str) -> str:
    """Patch anchors are re.escape() of literal pinned-source hunks."""
    return re.sub(r"\\(.)", r"\1", pattern, flags=re.DOTALL)


class VTP11HostGuardTests(unittest.TestCase):
    def _source(self, root: Path):
        for patch in patches():
            path = root / patch.path
            path.parent.mkdir(parents=True, exist_ok=True)
            chunks = [_unescape_literal(e.anchor) for e in patch.edits]
            path.write_text("\n".join(chunks) + "\n", encoding="utf-8")

    def test_package_is_a_narrow_opt_in(self):
        p = patches()
        self.assertEqual(len(p), 5)
        self.assertEqual(sum(len(f.edits) for f in p), 10)
        guards = [e.guard for f in p for e in f.edits]
        self.assertEqual(len(guards), len(set(guards)))
        for edit in (e for f in p for e in f.edits):
            self.assertIn("BC_VTP11_HOST_DOMAIN_", edit.guard)
            self.assertEqual(edit.expect_matches, 1)

    def test_applies_once_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._source(root)
            result = apply_all(patches(), root)
            self.assertTrue(all(p.ok for p in result), result)
            self.assertTrue(
                all(e.status == "applied" for p in result for e in p.results),
                result,
            )
            placed = {p.path: (root / p.path).read_bytes() for p in patches()}
            result2 = apply_all(patches(), root)
            self.assertTrue(
                all(e.status == "already-applied" for p in result2 for e in p.results),
                result2,
            )
            self.assertEqual(
                placed, {p.path: (root / p.path).read_bytes() for p in patches()}
            )
            planner = (root / "core/place/planner.cpp").read_text()
            self.assertIn("!host_backend && dev && o.host_unsafe_device_buffers", planner)
            self.assertIn("&device_ok, host_backend", planner)
            self.assertIn("&device_ok_, in_.host_backend", planner)
            self.assertIn("must be staged, not executed by a host kernel",
                          (root / "core/runtime/ctx.cpp").read_text())
            self.assertIn("!buf_host_[bh]",
                          (root / "core/runtime/issue.cpp").read_text())

    def test_missing_source_anchor_fails_closed_atomically(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._source(root)
            source = root / "core/runtime/ctx.cpp"
            before = source.read_text()
            needle = _unescape_literal(patches()[3].edits[1].anchor)
            self.assertIn(needle, before)
            source.write_text(before.replace(needle, "upstream changed"), encoding="utf-8")
            snapshot = {p.path: (root / p.path).read_bytes() for p in patches()}
            result = apply_all(patches(), root)
            self.assertTrue(any(not p.ok for p in result), result)
            self.assertEqual(
                snapshot, {p.path: (root / p.path).read_bytes() for p in patches()}
            )


if __name__ == "__main__":
    unittest.main()
