"""Card-free VTP09 gfx11 M17..64 IQ6R tiled row contract tests."""
from __future__ import annotations

import importlib.util
import pathlib
import unittest
from copy import deepcopy
from unittest.mock import patch

ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCE = ROOT / "lab/radiance/xtx27_iqr_m64_registry_check.py"
SPEC = importlib.util.spec_from_file_location("xtx27_iqr_m64_registry_check", SOURCE)
assert SPEC and SPEC.loader
q = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(q)


def fixture() -> dict[str, dict]:
    common = [("K", q.DIV, 256, None),
              ("N", q.DIV, 128, None),
              ("group", q.EQ, 128, None),
              ("dtype", q.IN, 0, "i8a8 i8a8h")]
    basic = {"op": "gemm_nt_q", "domain": 1,
             "launch": 1234, "scratch": 1235, "layout": 1236,
             "relayout": 1237, "unrelayout": 1238, "replaces": None}
    narrow = {**basic, "name": q.NARROW, "priority": 60,
              "constraints": [("M", q.LE, 64, None),
                              ("K", q.DIV, 256, None),
                              ("N", q.DIV, 16, None),
                              ("group", q.EQ, 128, None),
                              ("dtype", q.IN, 0, "i8a8 i8a8h")]}
    tiled = {**basic, "name": q.TILED, "priority": 40,
             "constraints": common}
    custom = {**basic, "name": q.CUSTOM, "priority": 61,
              "constraints": common + [("M", q.EQ, 64, None)]}
    early = {**basic, "name": q.EARLY, "priority": 61,
             "constraints": common + [("M", q.GE, 17, None),
                                      ("M", q.LE, 63, None),
                                      ("K", q.EQ, 5120, None),
                                      ("N", q.IN, 0, "5120 10240")]}
    return {q.NARROW: narrow, q.TILED: tiled, q.CUSTOM: custom, q.EARLY: early}


class M64RegistryTests(unittest.TestCase):
    def test_valid_model_shape_and_no_n16_orphan(self):
        result = q.validate(fixture(), active=True)
        self.assertTrue(result["narrow_fallback_remains"])
        self.assertGreaterEqual(result["tested_bands"], 25)
        for m, expected in ((1, q.NARROW), (8, q.NARROW), (16, q.NARROW),
                            (17, q.EARLY), (20, q.EARLY), (24, q.EARLY),
                            (28, q.EARLY), (31, q.EARLY), (32, q.EARLY),
                            (48, q.EARLY), (63, q.EARLY), (64, q.CUSTOM),
                            (65, q.TILED), (512, q.TILED)):
            self.assertEqual(
                q.selected(fixture(), m=m, n=10240, k=5120), expected,
            )
        self.assertEqual(q.selected(fixture(), m=64, n=16, k=5120), q.NARROW)

    def test_missing_row_or_disabled_rejected(self):
        rows = fixture()
        with self.assertRaisesRegex(ValueError, "missing"):
            q.validate(rows, active=False)
        rows.pop(q.EARLY)
        with self.assertRaisesRegex(ValueError, "missing"):
            q.validate(rows, active=True)

    def test_row_shape_drift_rejected(self):
        mutations = (
            (q.NARROW, "constraints", [("M", q.LE, 63, None)]),
            (q.TILED, "constraints", [("N", q.DIV, 16, None)]),
            (q.CUSTOM, "constraints", [("M", q.EQ, 64, None)]),
            (q.EARLY, "constraints", [("M", q.GE, 16, None)]),
            (q.EARLY, "priority", 59),
            (q.CUSTOM, "priority", 59),
            (q.TILED, "priority", 62),
            (q.CUSTOM, "op", "gemm_nt"),
        )
        for name, field, value in mutations:
            with self.subTest(name=name, field=field, value=value):
                rows = deepcopy(fixture())
                rows[name][field] = value
                with self.assertRaises(ValueError):
                    q.validate(rows, active=True)

    def test_m64_exact_tiled_hooks_not_just_same_names(self):
        for field in ("launch", "scratch", "layout", "relayout",
                      "unrelayout", "domain"):
            with self.subTest(field=field):
                rows = deepcopy(fixture())
                for row_name in (q.CUSTOM, q.EARLY):
                    with self.subTest(row=row_name):
                        rows = deepcopy(fixture())
                        rows[row_name][field] = 0
                        with self.assertRaises(ValueError):
                            q.validate(rows, active=True)

    def test_unsupported_shapes_do_not_gain_priority(self):
        rows = fixture()
        self.assertIsNone(q.selected(rows, m=64, n=15, k=5120))
        self.assertIsNone(q.selected(rows, m=64, n=10240, k=5000))
        self.assertEqual(q.selected(rows, m=64, n=48, k=5120), q.NARROW)
        self.assertEqual(q.selected(rows, m=64, n=128, k=5120), q.CUSTOM)
        self.assertEqual(q.selected(rows, m=16, n=128, k=5120), q.NARROW)
        self.assertEqual(q.selected(rows, m=17, n=128, k=5120), q.NARROW)
        self.assertEqual(q.selected(rows, m=32, n=128, k=5120), q.NARROW)
        self.assertEqual(q.selected(rows, m=17, n=5120, k=5120), q.EARLY)
        self.assertEqual(q.selected(rows, m=31, n=10240, k=5120), q.EARLY)
        self.assertEqual(q.selected(rows, m=17, n=8192, k=5120), q.NARROW)
        self.assertEqual(q.selected(rows, m=17, n=5120, k=1024), q.NARROW)
        self.assertEqual(q.selected(rows, m=17, n=512, k=1024), q.NARROW)
        self.assertEqual(q.selected(rows, m=63, n=48, k=5120), q.NARROW)

    def test_checked_ctypes_struct_constraint_offsets(self):
        self.assertEqual(q.CConstraint.ival.offset, 16)
        self.assertEqual(q.CKernel.n_constraints.offset, 64)
        self.assertEqual(q.CKernel.constraints.offset, 56)

    def test_no_overwrite_fails_before_dynamic_library_load(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            fake = root / "libr3.so"
            fake.write_bytes(b"fake")
            receipt = root / "receipt.json"
            receipt.write_text("{}")
            with (
                patch.object(q, "read_plugin", side_effect=AssertionError("loaded")),
                self.assertRaises(FileExistsError),
            ):
                q.main(["--plugin", str(fake), "--receipt", str(receipt)])


if __name__ == "__main__":
    unittest.main()
