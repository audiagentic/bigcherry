"""Offline negative tests of RDNA3 IQR code-image and hook audit."""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[2] / "lab/radiance/iq6r_v4_rdna3_preflight.py"
spec = importlib.util.spec_from_file_location("iq6r_v4_rdna3_preflight", SOURCE)
assert spec and spec.loader
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def fixture():
    names = ["r3_stub_begin", "r3_stub_end", *audit.REQUIRED]
    sym = "\n".join(f"{1000 + i * 16:x} T {name}" for i, name in enumerate(names))
    offload = "x.hipv4-amdgcn-amd-amdhsa--gfx1100"
    rows = "\n".join(f'{{ "{name}", "op", "family" }}' for name in audit.REQUIRED_ROWS)
    return sym, offload, rows


class RDNA3PreflightTests(unittest.TestCase):
    def test_complete_contract(self):
        report = audit.inspect(*fixture(), "# no exclusions")
        self.assertEqual(report["exported_iqr_hooks"], len(audit.REQUIRED))
        self.assertFalse(report["numeric_tests_passed"])

    def test_rejects_missing_image(self):
        sym, _, rows = fixture()
        with self.assertRaisesRegex(ValueError, "gfx1100"):
            audit.inspect(sym, "a.hipv4-amdgcn-amd-amdhsa--gfx1201", rows, "")

    def test_rejects_missing_hook(self):
        sym, offload, rows = fixture()
        sym = "\n".join(line for line in sym.splitlines()
                        if not line.endswith("r4d_rad_gemm_iqr_tiled"))
        with self.assertRaisesRegex(ValueError, "missing exported"):
            audit.inspect(sym, offload, rows, "")

    def test_rejects_bad_stub_bounds(self):
        sym, offload, rows = fixture()
        sym = sym.replace("3e8 T r3_stub_begin", "8000 T r3_stub_begin")
        with self.assertRaisesRegex(ValueError, "malformed stub"):
            audit.inspect(sym, offload, rows, "")

    def test_rejects_missing_row(self):
        sym, offload, rows = fixture()
        rows = rows.replace('{ "gemm_iqr_tiled", "op", "family" }', "")
        with self.assertRaisesRegex(ValueError, "required upstream IQR"):
            audit.inspect(sym, offload, rows, "")

    def test_rejects_excluded_unit(self):
        with self.assertRaisesRegex(ValueError, "excluded"):
            audit.inspect(*fixture(), "r4d_gemm_iqr\n")


if __name__ == "__main__":
    unittest.main()
