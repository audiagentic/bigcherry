"""Regression tests for the retained lab analyzer's activation stop condition."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "lab/gp11-replay-bench/analyse.py"
HEADER = "start rounds=2 arms=control:abc:no:replay winners:abc:yes:replay model=m devices=0,1\n"


class ReplayBenchActivationTests(unittest.TestCase):
    def run_report(self, content):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "run.log"
            log.write_text(content, encoding="utf-8")
            return subprocess.run(
                [sys.executable, str(SCRIPT), str(log)],
                capture_output=True, text=True, check=False,
            )

    def assert_rejected(self, content):
        result = self.run_report(content)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("ACTIVATION EVIDENCE FAILED", result.stdout)
        self.assertNotIn("baseline=", result.stdout)

    def test_swallowed_cache_logs_do_not_hide_intended_cache_arm(self):
        self.assert_rejected(HEADER + "round=1 pos=2 arm=winners tg512_tps: 100\n"
                             "round=1 pos=2 arm=winners cacheload: none\n")

    def test_one_active_cell_cannot_validate_another_cell(self):
        self.assert_rejected(HEADER + "round=1 pos=2 arm=winners tg512_tps: 100\n"
                             "round=1 pos=2 arm=winners replaystat: tuned=50\n"
                             "round=2 pos=1 arm=winners tg512_tps: 100\n")

    def test_zero_launches_and_exact_hits_are_rejected(self):
        self.assert_rejected(HEADER + "round=1 pos=2 arm=winners tg512_tps: 100\n"
                             "round=1 pos=2 arm=winners replaystat: exact=100 tuned=0\n")

    def test_missing_arm_declaration_is_rejected(self):
        self.assert_rejected("round=1 pos=2 arm=winners tg512_tps: 100\n")

    def test_shutdown_warning_blocks_report(self):
        self.assert_rejected(HEADER + "round=1 pos=2 arm=winners tg512_tps: 100\n"
                             "round=1 pos=2 arm=winners replaystat: tuned=50\n"
                             "round=1 pos=2 arm=winners WARN shutdown-endpoint-unavailable\n")

    def test_positive_launch_evidence_allows_activation_check(self):
        result = self.run_report(HEADER + "round=1 pos=1 arm=control tg512_tps: 100\n"
                                 "round=1 pos=2 arm=winners tg512_tps: 101\n"
                                 "round=1 pos=2 arm=winners replaystat: tuned=50\n")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("baseline=control", result.stdout)


if __name__ == "__main__":
    unittest.main()
