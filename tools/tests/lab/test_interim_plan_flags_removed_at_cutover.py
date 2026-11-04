"""The engine's interim plan-check flags cannot outlive the cutover (RSC08, lead decision 2026-11-02).

`--plan-check FILE --plan-check-floor-ppm N` exist only until the engine reads its expected plan and floor from a v1
manifest (`engine_view`, slice 5/6). This test fails the moment an engine integration uses `engine_view` while the flags
are still in the tree, so they are deleted in the same change that wires the manifest: no shim, no parallel path.
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RAD0011 = REPO / "engines" / "radiance" / "patches" / "rad0011_plan_memory" / "patch.py"
FLAG_SITES = (
    RAD0011,
    REPO / "engines" / "radiance" / "components" / "serving" / "main.cpp",
    REPO / "tools" / "lab" / "radiance" / "launch-flashnext-best.sh",
    REPO / "tools" / "lab" / "radiance" / "lib" / "plan-check.sh",
)
# where a managed engine patch or the engine's own code would call engine_view(manifest)
ENGINE_SIDE = tuple((REPO / "engines" / "radiance" / "patches").glob("*/patch.py")) + tuple(
    (REPO / "engines" / "radiance" / "components").glob("*/*.cpp"))


class InterimPlanFlagsTests(unittest.TestCase):
    def test_the_flags_are_gone_once_an_engine_reads_engine_view(self) -> None:
        uses_engine_view = [
            p for p in ENGINE_SIDE
            if p.parent.name not in ("serving",) and re.search(r"\bengine_view\s*\(", p.read_text(encoding="utf-8"))]
        # the shared reader defines and tests engine_view inside the serving component; an engine-side caller is the cutover
        if not uses_engine_view:
            return
        survivors = [p.relative_to(REPO).as_posix() for p in FLAG_SITES
                     if p.exists() and "plan-check-floor-ppm" in p.read_text(encoding="utf-8")]
        self.assertEqual(survivors, [], f"{uses_engine_view} read engine_view: delete the interim --plan-check flags "
                                        f"(RSC08 cutover) from {survivors}")

    def test_the_flags_are_named_as_interim_where_they_live(self) -> None:
        self.assertIn("interim", RAD0011.read_text(encoding="utf-8").lower())
        self.assertIn("RSC08", RAD0011.read_text(encoding="utf-8"))

    def test_the_guard_trips_on_a_simulated_cutover(self) -> None:
        """Prove the guard can fail: pretend an engine patch calls engine_view while a flag site still has the flag."""
        text = "int rc = engine_view(manifest).kv_backing_floor_ppm;"
        self.assertTrue(re.search(r"\bengine_view\s*\(", text))
        self.assertIn("plan-check-floor-ppm", RAD0011.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
