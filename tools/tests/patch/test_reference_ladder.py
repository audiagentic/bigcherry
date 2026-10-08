"""PVPS03: reference ladder (stock / base / validated / validated+patch)."""

from __future__ import annotations

import subprocess
import tempfile
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
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.model = self.tmp / "m.gguf"
        self.model.write_bytes(b"gguf")

    def _ladder(self, **kwargs):
        kwargs.setdefault("runner", self._runner)
        kwargs.setdefault("runtime_args", ())
        return ladder.run_reference_ladder(
            arms=_arms(self.root), model=self.model, cache_dir=self.tmp / "cache",
            shared_arms=frozenset({"stock", "base", "validated"}), device_key="gfx1100:0", **kwargs,
        )

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
        payload = self._ladder(workloads=("decode", "prefill"))
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
        self._ladder(workloads=("decode",), rounds_per_arm=2)
        firsts = [self.calls[i] for i in range(0, len(self.calls), 3)]
        self.assertEqual(firsts[:3], ["stock", "base", "subject"])

    def test_failed_runs_are_counted_not_averaged(self) -> None:
        def failing(command):
            return subprocess.CompletedProcess(command, 1, stdout="", stderr="boom")

        payload = self._ladder(runner=failing, workloads=("decode",))
        self.assertEqual(payload["metrics"]["tg128"]["failed_runs"], 18)  # 6 rounds x 3 distinct binaries
        self.assertEqual(payload["metrics"]["tg128"]["mean"], {})

    def test_shared_arms_are_measured_once_then_reused(self) -> None:
        first = self._ladder(workloads=("decode", "prefill"))
        self.assertEqual(first["cached_arms"], [])
        self.calls.clear()
        second = self._ladder(workloads=("decode", "prefill"))
        # Only the subject is re-measured, with the same sample count as before
        # (3 distinct binaries x 2 rotations = 6 rounds).
        self.assertEqual(self.calls, ["subject"] * 6)
        self.assertEqual(second["cached_arms"], ["base", "stock", "validated"])
        self.assertEqual(second["metrics"]["tg128"]["mean"], first["metrics"]["tg128"]["mean"])

    def test_failed_shared_measurement_is_not_cached(self) -> None:
        def failing(command):
            return subprocess.CompletedProcess(command, 1, stdout="", stderr="boom")

        self._ladder(runner=failing, workloads=("decode",))
        self.calls.clear()
        self._ladder(workloads=("decode",))
        self.assertEqual(set(self.calls), {"stock", "base", "subject"})

    def test_in_place_binary_rebuild_invalidates_the_cache(self) -> None:
        # GPT review req_6c90e1ebba83464a: a build root is content-addressed
        # by SOURCE, not by binary bytes -- an in-place rebuild at the same
        # binary_dir (e.g. ccache regenerating different code without the
        # tree's identity changing) must not serve the old cached numbers.
        real_root = self.tmp / "real-arms"
        for name in ("stock", "subject"):
            (real_root / name).mkdir(parents=True)
            (real_root / name / "llama-bench").write_bytes(b"v1")
        arms = {"stock": real_root / "stock", "base": real_root / "stock",
                "validated": real_root / "stock", "validated+patch": real_root / "subject"}
        first = ladder.run_reference_ladder(
            arms=arms, model=self.model, runner=self._runner, cache_dir=self.tmp / "cache",
            shared_arms=frozenset({"stock", "base", "validated"}), device_key="gfx1100:0", runtime_args=(),
            workloads=("decode",),
        )
        self.assertEqual(first["cached_arms"], [])
        self.calls.clear()
        # Simulate an in-place rebuild: same path, new content/mtime.
        (real_root / "stock" / "llama-bench").write_bytes(b"v2-different-build")
        second = ladder.run_reference_ladder(
            arms=arms, model=self.model, runner=self._runner, cache_dir=self.tmp / "cache",
            shared_arms=frozenset({"stock", "base", "validated"}), device_key="gfx1100:0", runtime_args=(),
            workloads=("decode",),
        )
        self.assertEqual(second["cached_arms"], [], "a rebuilt binary must be a cache miss, not stale-served")
        self.assertIn("stock", self.calls)

    def test_runtime_args_reach_the_command_and_the_cache_key(self):
        seen: list[list[str]] = []

        def runner(command):
            seen.append(command)
            return self._runner(command)

        self._ladder(runner=runner, runtime_args=("-sm", "tensor"))
        self.assertTrue(all(c[-2:] == ["-sm", "tensor"] for c in seen))


if __name__ == "__main__":
    unittest.main()
