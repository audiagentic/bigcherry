"""PVPS03: reference ladder (stock / base / validated / validated+patch)."""

from __future__ import annotations

import subprocess
import unittest
from pathlib import Path

from bigcherry.patch.campaign import ladder

SPEED = {"stock": 100.0, "base": 105.0, "subject": 110.0}


def _arms(root: Path) -> dict[str, Path]:
    # base and validated share one binary: the promoted set is empty.
    return {
        "stock": root / "stock",
        "base": root / "base",
        "validated": root / "base",
        "validated+patch": root / "subject",
    }


class ReferenceLadderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path("/ladder-test")
        self.calls: list[str] = []

    def _runner(self, command: list[str]) -> subprocess.CompletedProcess:
        arm = Path(command[0]).parent.name
        self.calls.append(arm)
        metric = "tg128" if "-n" in command and "128" in command else "pp512"
        return subprocess.CompletedProcess(command, 0, stdout=f"| {metric} | {SPEED[arm]:.2f} ± 0.1 |", stderr="")

    def test_shared_binary_is_measured_once_and_reported_under_every_name(self) -> None:
        payload = ladder.run_reference_ladder(
            arms=_arms(self.root), model=Path("m.gguf"), runner=self._runner, workloads=("decode",)
        )
        # 3 distinct binaries x 2 rounds each = 6 rounds x 3 arms.
        self.assertEqual(len(self.calls), 18)
        self.assertEqual(set(self.calls), {"stock", "base", "subject"})
        tg = payload["metrics"]["tg128"]
        self.assertEqual(tg["mean"]["base"], tg["mean"]["validated"])
        self.assertAlmostEqual(tg["pct_vs_stock"]["validated+patch"], 10.0)
        self.assertEqual(payload["shared_binaries"], [["base", "validated"]])
        self.assertTrue(payload["reference_only"])

    def test_order_rotates_every_round(self) -> None:
        ladder.run_reference_ladder(
            arms=_arms(self.root), model=Path("m.gguf"), runner=self._runner, workloads=("decode",)
        )
        firsts = [self.calls[i] for i in range(0, len(self.calls), 3)]
        self.assertEqual(firsts[:3], ["stock", "base", "subject"])

    def test_failed_runs_are_counted_not_averaged(self) -> None:
        def failing(command):
            return subprocess.CompletedProcess(command, 1, stdout="", stderr="boom")

        payload = ladder.run_reference_ladder(
            arms=_arms(self.root), model=Path("m.gguf"), runner=failing, workloads=("decode",)
        )
        self.assertEqual(payload["metrics"]["tg128"]["failed_runs"], 18)
        self.assertEqual(payload["metrics"]["tg128"]["mean"], {})


if __name__ == "__main__":
    unittest.main()
