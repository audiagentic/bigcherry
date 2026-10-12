"""RS06 tests for tools/bigcherry/focal_source_plans.py (patch-system
PA02). Runbook section 59 test list: no dependencies, single dependency,
transitive dependencies, dependency already in baseline, deterministic
order, focal already in baseline, another baseline patch depends on
focal, conflict introduced by focal, rejected dependency.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.patch import source_plans as fsp # noqa: E402
from bigcherry.patch import registry as patch_registry # noqa: E402


def _flat_module(state: str, extra: str = "") -> str:
    return f'STATE = "{state}"\n{extra}PATCHES = []\n'


class FocalComparisonTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="bigcherry-fsp-")
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name) / "patches"
        self.root.mkdir()

    def _add_patch(self, patch_id: str, state: str = "untested", extra: str = "") -> None:
        (self.root / f"{patch_id}.py").write_text(
            _flat_module(state, extra), encoding="utf-8"
        )

    def _registry(self) -> patch_registry.PatchRegistry:
        return patch_registry.load_registry(self.root)

    def test_no_dependencies(self) -> None:
        self._add_patch("0100_a")
        self._add_patch("0200_b")
        self._add_patch("0300_x")
        result = fsp.build_focal_comparison(
            "0300_x", ["0100_a", "0200_b"], registry=self._registry()
        )
        self.assertFalse(result.is_blocked)
        self.assertEqual(result.control, ("0100_a", "0200_b"))
        self.assertEqual(result.subject, ("0100_a", "0200_b", "0300_x"))
        self.assertEqual(result.prerequisites, ())
        self.assertEqual(result.stock, ())
        self.assertEqual(set(result.subject) - set(result.control), {"0300_x"})

    def test_single_dependency(self) -> None:
        self._add_patch("0100_a")
        self._add_patch("0200_p")
        self._add_patch("0300_x", extra='REQUIRES = ("0200_p",)\n')
        result = fsp.build_focal_comparison(
            "0300_x", ["0100_a"], registry=self._registry()
        )
        self.assertFalse(result.is_blocked)
        self.assertEqual(result.prerequisites, ("0200_p",))
        self.assertEqual(result.control, ("0100_a", "0200_p"))
        self.assertEqual(result.subject, ("0100_a", "0200_p", "0300_x"))

    def test_transitive_dependencies(self) -> None:
        self._add_patch("0100_a")
        self._add_patch("0200_b", extra='REQUIRES = ("0100_a",)\n')
        self._add_patch("0300_x", extra='REQUIRES = ("0200_b",)\n')
        result = fsp.build_focal_comparison("0300_x", [], registry=self._registry())
        self.assertFalse(result.is_blocked)
        self.assertEqual(result.prerequisites, ("0100_a", "0200_b"))
        self.assertEqual(result.control, ("0100_a", "0200_b"))
        self.assertEqual(result.subject, ("0100_a", "0200_b", "0300_x"))

    def test_dependency_already_in_baseline(self) -> None:
        self._add_patch("0100_a")
        self._add_patch("0200_x", extra='REQUIRES = ("0100_a",)\n')
        result = fsp.build_focal_comparison(
            "0200_x", ["0100_a"], registry=self._registry()
        )
        self.assertFalse(result.is_blocked)
        # The prerequisite is already carried by the baseline: control is
        # exactly the baseline, subject adds only the focal.
        self.assertEqual(result.control, ("0100_a",))
        self.assertEqual(result.subject, ("0100_a", "0200_x"))

    def test_dependency_order_is_deterministic(self) -> None:
        self._add_patch("0100_a")
        self._add_patch("0101_a2", extra='REQUIRES = ("0100_a",)\n')
        self._add_patch("0102_a3", extra='REQUIRES = ("0101_a2",)\n')
        self._add_patch("0200_x", extra='REQUIRES = ("0102_a3",)\n')
        first = fsp.build_focal_comparison("0200_x", [], registry=self._registry())
        second = fsp.build_focal_comparison("0200_x", [], registry=self._registry())
        self.assertEqual(first, second)
        self.assertEqual(first.prerequisites, ("0100_a", "0101_a2", "0102_a3"))

    def test_focal_already_in_baseline(self) -> None:
        self._add_patch("0100_a")
        self._add_patch("0200_x")
        result = fsp.build_focal_comparison(
            "0200_x", ["0100_a", "0200_x"], registry=self._registry()
        )
        self.assertTrue(result.is_blocked)
        self.assertIn("already in the baseline", result.blocked_reason)
        self.assertEqual(result.control, ())
        self.assertEqual(result.subject, ())

    def test_baseline_patch_depends_on_focal(self) -> None:
        self._add_patch("0100_x")
        self._add_patch("0200_y", extra='REQUIRES = ("0100_x",)\n')
        result = fsp.build_focal_comparison(
            "0100_x", ["0200_y"], registry=self._registry()
        )
        self.assertTrue(result.is_blocked)
        self.assertIn("depends on the focal", result.blocked_reason)

    def test_conflict_introduced_by_focal(self) -> None:
        self._add_patch("0100_a")
        self._add_patch("0200_x", extra='CONFLICTS = ("0100_a",)\n')
        result = fsp.build_focal_comparison(
            "0200_x", ["0100_a"], registry=self._registry()
        )
        self.assertTrue(result.is_blocked)
        self.assertIn("conflicts", result.blocked_reason)
        self.assertIn("0100_a", result.blocked_reason)

    def test_rejected_dependency(self) -> None:
        self._add_patch("0100_p", state="rejected")
        self._add_patch("0200_x", extra='REQUIRES = ("0100_p",)\n')
        result = fsp.build_focal_comparison("0200_x", [], registry=self._registry())
        self.assertTrue(result.is_blocked)
        self.assertIn("rejected", result.blocked_reason)
        self.assertIn("0100_p", result.blocked_reason)

    def test_unknown_focal_raises(self) -> None:
        self._add_patch("0100_a")
        with self.assertRaises(fsp.FocalComparisonError):
            fsp.build_focal_comparison(
                "9999_missing", ["0100_a"], registry=self._registry()
            )


if __name__ == "__main__":
    unittest.main()
