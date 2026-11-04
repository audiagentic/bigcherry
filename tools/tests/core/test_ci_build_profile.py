"""BU02: Ninja timing profile extraction is read-only and deterministic."""
from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[3] / ".github" / "scripts" / "ci_build_profile.py"
spec = importlib.util.spec_from_file_location("ci_build_profile", SCRIPT)
assert spec and spec.loader
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)


class BuildProfileTests(unittest.TestCase):
    def test_missing_directory_has_explanatory_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = profile.hip_durations(Path(tmp) / "absent")
        self.assertEqual(rows, [])
        self.assertIn("No HIP compilations recorded", profile.render(rows))

    def test_hip_timing_sorted_and_other_targets_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "store" / "plan" / "libr3"
            first.mkdir(parents=True)
            (first / ".ninja_log").write_text(
                "# ninja log v5\n"
                "0\t2000\t0\tCMakeFiles/libr3.dir/units/a.hip.o\ta1\n"
                "0\t9000\t0\tCMakeFiles/libr3.dir/units/b.hip.o\tb1\n"
                "0\t100000\t0\tCMakeFiles/libr3.dir/src/c.cpp.o\tc1\n"
                "garbage\n"
                "8\t3\t0\tbroken.hip.o\th1\n",
                encoding="utf-8",
            )
            rows = profile.hip_durations(root)
        self.assertEqual([duration for duration, _ in rows], [9000, 2000])
        self.assertTrue(rows[0][1].endswith("b.hip.o"))
        self.assertIn("9.0s", profile.render(rows))
        self.assertIn("not wall time", profile.render(rows))

    def test_logs_across_components_are_included(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for component, elapsed in (("libr3", 300), ("libr4d", 500)):
                path = root / component
                path.mkdir()
                (path / ".ninja_log").write_text(
                    f"# ninja log v5\n1\t{elapsed + 1}\t0\tunit.hip.o\t123\n",
                    encoding="utf-8",
                )
            rows = profile.hip_durations(root)
        self.assertEqual(rows, [(500, "libr4d/unit.hip.o"), (300, "libr3/unit.hip.o")])


if __name__ == "__main__":
    unittest.main()
