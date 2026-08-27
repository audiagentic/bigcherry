"""PROF01/HI132: offline tests for control-block measurement."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.profiling.control import run_control_block  # noqa: E402


class RunControlBlockTests(unittest.TestCase):
    def test_computes_mean_and_stddev_from_wall_clock(self):
        fake_runner = MagicMock()
        fake_runner.run_completion = MagicMock(return_value={})

        import bigcherry.profiling.control as control_mod

        times = iter([0.0, 1.0, 1.0, 2.0])  # two reps, each 1.0s elapsed

        def fake_monotonic():
            return next(times)

        original = control_mod.time.monotonic
        control_mod.time.monotonic = fake_monotonic
        try:
            block = run_control_block(
                runner=fake_runner, label="A", reps=2,
                prompt="hi", n_predict=100,
            )
        finally:
            control_mod.time.monotonic = original

        self.assertEqual(block.label, "A")
        self.assertEqual(block.reps, 2)
        self.assertEqual(block.tg_tps_values, (100.0, 100.0))
        self.assertAlmostEqual(block.tg_tps_mean, 100.0)
        self.assertAlmostEqual(block.tg_tps_stddev, 0.0)
        self.assertEqual(fake_runner.run_completion.call_count, 2)

    def test_zero_reps_gives_empty_block(self):
        fake_runner = MagicMock()
        block = run_control_block(
            runner=fake_runner, label="A", reps=0, prompt="hi", n_predict=100,
        )
        self.assertEqual(block.tg_tps_values, ())
        self.assertEqual(block.tg_tps_mean, 0.0)
        self.assertEqual(block.tg_tps_stddev, 0.0)


if __name__ == "__main__":
    unittest.main()
