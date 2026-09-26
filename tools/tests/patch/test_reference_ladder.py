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
        rows = []
        if command[command.index("-p") + 1] != "0":
            rows.append(f"| pp512 | {SPEED[arm] * 30:.2f} ± 1 |")
        if command[command.index("-n") + 1] != "0":
            rows.append(f"| tg128 | {SPEED[arm]:.2f} ± 0.1 |")
        return subprocess.CompletedProcess(command, 0, stdout="\n".join(rows), stderr="")

    def test_shared_binary_is_measured_once_and_reported_under_every_name(self) -> None:
        payload = ladder.run_reference_ladder(
            arms=_arms(self.root), model=Path("m.gguf"), runner=self._runner, workloads=("decode", "prefill")
        )
        # 3 distinct binaries x 2 rotations = 6 rounds x 3 arms; ONE call measures both workloads.
        self.assertEqual(len(self.calls), 18)
        self.assertAlmostEqual(payload["metrics"]["pp512"]["pct_vs_stock"]["validated+patch"], 10.0)
        self.assertEqual(set(self.calls), {"stock", "base", "subject"})
        tg = payload["metrics"]["tg128"]
        self.assertEqual(tg["mean"]["base"], tg["mean"]["validated"])
        self.assertAlmostEqual(tg["pct_vs_stock"]["validated+patch"], 10.0)
        self.assertEqual(payload["shared_binaries"], [["base", "validated"]])
        self.assertTrue(payload["reference_only"])

    def test_order_rotates_every_round(self) -> None:
        ladder.run_reference_ladder(
            arms=_arms(self.root), model=Path("m.gguf"), runner=self._runner, workloads=("decode",),
            rounds_per_arm=2,
        )
        firsts = [self.calls[i] for i in range(0, len(self.calls), 3)]
        self.assertEqual(firsts[:3], ["stock", "base", "subject"])

    def test_failed_runs_are_counted_not_averaged(self) -> None:
        def failing(command):
            return subprocess.CompletedProcess(command, 1, stdout="", stderr="boom")

        payload = ladder.run_reference_ladder(
            arms=_arms(self.root), model=Path("m.gguf"), runner=failing, workloads=("decode",)
        )
        self.assertEqual(payload["metrics"]["tg128"]["failed_runs"], 18)  # 6 rounds x 3 distinct binaries
        self.assertEqual(payload["metrics"]["tg128"]["mean"], {})


if __name__ == "__main__":
    unittest.main()
