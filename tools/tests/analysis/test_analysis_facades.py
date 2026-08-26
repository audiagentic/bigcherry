"""RA37/TR12 guard against reintroducing removed root-level analysis facades.

All nine historical root-level facades are now removed because every live
consumer imports the canonical bigcherry.analysis modules directly.
"""

from __future__ import annotations

import unittest
from pathlib import Path


# Guarded so a reintroduced stray root module would be caught here rather
# than silently shadowing the canonical bigcherry.analysis surface.
REMOVED_FACADES = (
    "bigcherry.bandit_simulator",
    "bigcherry.candidate_binary_size",
    "bigcherry.analyze_gaps",
    "bigcherry.compare_tunes",
    "bigcherry.impact",
    "bigcherry.kernel_fraction",
    "bigcherry.report",
    "bigcherry.resource_report",
    "bigcherry.symbol_map",
)


class AnalysisFacadeParityTests(unittest.TestCase):
    def test_removed_facades_are_gone(self):
        tools_root = Path(__file__).resolve().parents[2] / "bigcherry"
        for removed_path in REMOVED_FACADES:
            with self.subTest(removed=removed_path):
                filename = removed_path.rsplit(".", 1)[1] + ".py"
                self.assertFalse(
                    (tools_root / filename).exists(),
                    f"root facade {filename} was removed by TR12 and must not return",
                )


if __name__ == "__main__":
    unittest.main()
