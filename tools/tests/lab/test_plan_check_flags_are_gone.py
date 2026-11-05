"""The engine's interim plan-check flags are gone and cannot come back (RSC08 S3, lead decision 2026-11-02).

`--plan-check FILE --plan-check-floor-ppm N` (rad0011) and bc-serve's `--plan-check-args`, `--plan-store`, `--plan-key` existed only
until the engine read its expected plan and floor from a v1 manifest. The engine now takes both from the measured manifest it is
started from (rad0009's bc_serving_expected_plan, from engine_view), bc-serve writes the measured manifest from a measuring
start's plan (`--write-manifest ... --measured-plan FILE`), and no flag can hand the engine a plan. This test fails if any of the
retired names reappears in the code that produced or consumed them: no shim, no parallel path.
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RETIRED = ("--plan-check", "plan-check-floor-ppm", "--plan-check-args", "--plan-store", "--plan-key", "plan_check_args",
           "--expect-key", "plan_check_flags")
CODE = (
    *(REPO / "engines" / "radiance" / "patches").glob("rad00*/patch.py"),
    *(REPO / "engines" / "radiance" / "patches").glob("rad001*/patch.py"),
    *(REPO / "engines" / "radiance" / "components" / "serving").glob("*.cpp"),
    *(REPO / "engines" / "radiance" / "components" / "serving").glob("*.h"),
    *(REPO / "engines" / "radiance" / "components" / "serving" / "v1").glob("*.cpp"),   # the refusals' hints name the remedy
    *(REPO / "engines" / "radiance" / "components" / "serving" / "v1").glob("*.h"),
    *(REPO / "tools" / "lab" / "radiance").rglob("*.sh"),
    *(REPO / "tools" / "bigcherry").rglob("*.py"),
)
# the card-free tests name the retired flags to show they are refused; the contract examples never carry them
ALLOWED_MENTIONS = {"plan_flags_test.cpp", "manifest_argv_test.cpp"}


class PlanCheckFlagsAreGoneTests(unittest.TestCase):
    def test_no_retired_name_survives_in_the_code(self) -> None:
        survivors = [f"{p.relative_to(REPO)}: {name}" for p in CODE if p.is_file() and p.name not in ALLOWED_MENTIONS
                     for name in RETIRED if name in p.read_text(encoding="utf-8")]
        self.assertEqual(survivors, [])

    def test_the_engine_checks_the_plan_its_manifest_carries(self) -> None:
        rad0009 = (REPO / "engines/radiance/patches/rad0009_serving_manifest/patch.py").read_text(encoding="utf-8")
        rad0011 = (REPO / "engines/radiance/patches/rad0011_plan_memory/patch.py").read_text(encoding="utf-8")
        self.assertIn("bc_serving_expected_plan", rad0009)
        self.assertRegex(rad0009, r"engine_view\(manifest\.value\(\)\)")
        self.assertIn("bc_serving_expected_plan(&expected, &floor_ppm)", rad0011)

    def test_the_guard_can_fail(self) -> None:
        self.assertTrue(any(name in "bc-serve --plan-store --plan-file F" for name in RETIRED))
        self.assertFalse(any(name in "bc-serve --write-manifest OUT --measured-plan F" for name in RETIRED))


if __name__ == "__main__":
    unittest.main()
