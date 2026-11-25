"""Unit harness: the card-free parts (resource diff, hash comparison, replay estimator, selftest edit, stub generator)."""
import importlib.util
import sys
import unittest
from pathlib import Path

PATH = Path(__file__).resolve().parents[2] / "lab" / "radiance" / "unit_harness.py"
spec = importlib.util.spec_from_file_location("unit_harness", PATH)
uh = importlib.util.module_from_spec(spec)
sys.modules["unit_harness"] = uh
spec.loader.exec_module(uh)

REC = dict.fromkeys(uh.RESOURCE_KEYS, 0)


class HarnessTests(unittest.TestCase):
    def test_resource_diff(self):
        ref = {"a_iqr": {**REC, "vgpr_count": 100}, "b_iqr": dict(REC), "c_other": dict(REC)}
        cand = {"a_iqr": {**REC, "vgpr_count": 101}, "new_iqr": dict(REC), "c_other": dict(REC)}
        import re
        diff = uh.diff_records(ref, cand, re.compile("iqr"))
        self.assertEqual(diff["added"], ["new_iqr"])
        self.assertEqual(diff["removed"], ["b_iqr"])
        self.assertEqual(diff["changed"], {"a_iqr": {"vgpr_count": [100, 101]}})
        self.assertEqual(diff["unchanged"], 1)

    def test_hash_comparison(self):
        got = uh.compare_hashes({"x": "1", "y": "2", "z": "3"}, {"x": "1", "y": "9", "w": "4"})
        self.assertEqual((got["compared"], got["differ"], got["missing_in_candidate"], got["only_in_candidate"]),
                         (2, ["y"], ["z"], ["w"]))

    def test_estimator_ignores_slow_mode_windows(self):
        self.assertFalse(uh.converged([300.0, 301.0, 300.5], 0.015))      # too few windows
        self.assertTrue(uh.converged([300.0, 150.0, 151.0, 300.0], 0.015))
        self.assertFalse(uh.converged([300.0, 150.0, 190.0, 300.0], 0.015))
        self.assertEqual(uh.estimate([300.0, 150.0, 151.0]), 150.0)

    def test_classify(self):
        self.assertEqual([uh.classify(r, 0.04) for r in (0.5, 1.0, 1.1)], ["win", "same", "regress"])

    def test_selftest_edit_needs_both_anchors_once(self):
        text = uh.HASH_ANCHOR + uh.CASES_ANCHOR
        edited = uh.inject_selftest(text)
        self.assertIn("UNITHASH", edited)
        self.assertIn("UH_HASH_CASES", edited)
        with self.assertRaises(uh.HarnessError):
            uh.inject_selftest(text + uh.CASES_ANCHOR)
        with self.assertRaises(uh.HarnessError):
            uh.inject_selftest("nothing")

    def test_families_are_well_formed(self):
        for name, fam in uh.families().items():
            self.assertTrue(fam["units"], name)
            self.assertIn("hash", fam)
            self.assertIn("replay", fam)
        cases = uh.hash_cases(uh.families()["iqr"]["hash"]).split(",")
        self.assertIn("5376:2560:129", cases)
        self.assertEqual(len(cases), len(set(cases)))


if __name__ == "__main__":
    unittest.main()
