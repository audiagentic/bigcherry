"""RHP16 calibration launcher guards; entirely card-free."""
from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
LAUNCHER = REPO / "tools/lab/radiance/launch-flashnext-best.sh"


class CalibrationLauncherTests(unittest.TestCase):
    def run_start(self, *flags: str) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as root:
            return subprocess.run(
                ["bash", str(LAUNCHER), "start", *flags],
                cwd=REPO,
                env=dict(os.environ, BC_WORK_ROOT=root),
                capture_output=True, text=True, timeout=10,
            )

    def workloads(self, root: str) -> tuple[Path, Path, Path]:
        base = Path(root)
        policy, a, b = (base / n for n in ("policy.json", "workload-a.txt", "workload-b.txt"))
        policy.write_text('{"cadence_ns": 5000000000}')
        a.write_text("Distinct filler traffic\n")
        b.write_text("Documentation queries with a different hot set\n")
        return policy, a, b

    def args(self, policy: Path, a: Path, b: Path) -> tuple[str, ...]:
        return ("--calibrate", "--live-policy", str(policy),
                "--calibration-prompts-a", str(a), "--calibration-prompts-b", str(b))

    def test_calibrate_requires_policy(self) -> None:
        r = self.run_start("--calibrate")
        self.assertEqual(r.returncode, 2)
        self.assertIn("--live-policy", r.stderr)

    def test_calibrate_requires_both_workloads(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            policy, a, b = self.workloads(root)
            for opts in (("--calibrate", "--live-policy", str(policy)),
                         ("--calibrate", "--live-policy", str(policy),
                          "--calibration-prompts-a", str(a))):
                r = self.run_start(*opts)
                self.assertEqual(r.returncode, 2)
                self.assertIn("--calibration-prompts-a and --calibration-prompts-b", r.stderr)

    def test_rejects_identical_workloads(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            policy, a, b = self.workloads(root)
            b.write_bytes(a.read_bytes())
            r = self.run_start(*self.args(policy, a, b))
            self.assertEqual(r.returncode, 2)
            self.assertIn("distinct heat-shifting workloads", r.stderr)

    def test_rejects_blank_workload_in_either_position(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            policy, a, b = self.workloads(root)
            for blank in (a, b):
                blank.write_text("  \n\t\n")
                r = self.run_start(*self.args(policy, a, b))
                self.assertEqual(r.returncode, 2)
                self.assertIn("nonblank", r.stderr)
                if blank == a:
                    a.write_text("Filler query\n")

    def test_calibrate_refuses_experiment(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            policy, a, b = self.workloads(root)
            r = self.run_start(*self.args(policy, a, b),
                               "--experiment", str(Path(root) / "experiment.json"))
            self.assertEqual(r.returncode, 2)
            self.assertIn("takes no --experiment", r.stderr)

    def test_calibrate_rejects_disabled_measurement_source(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            policy, a, b = self.workloads(root)
            r = self.run_start(*self.args(policy, a, b), "--record-heat", "off")
            self.assertEqual(r.returncode, 2)
            self.assertIn("histogram", r.stderr)

    def test_calibrate_requires_active_policy_hooks(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            policy, a, b = self.workloads(root)
            policy.write_text("{}")
            r = self.run_start(*self.args(policy, a, b))
            self.assertEqual(r.returncode, 2)
            self.assertIn("present = active", r.stderr)

    def test_calibrate_refuses_enabled_member(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            policy, a, b = self.workloads(root)
            policy.write_text('{"enabled": true, "cadence_ns": 5000000000}')
            r = self.run_start(*self.args(policy, a, b))
            self.assertEqual(r.returncode, 2)
            self.assertIn("no 'enabled' member", r.stderr)

    def test_both_orders_before_accepting_marker(self) -> None:
        # Inspect the card-free shell control path: a single A->B sequence
        # would not exercise the reversed direction of the hot-set shift.
        source = LAUNCHER.read_text()
        body = source.split('if [ "$CALIBRATE" = 1 ]; then')[-1]
        self.assertIn('for prompts in "$CALIBRATION_PROMPTS_A" "$CALIBRATION_PROMPTS_B" \\\n                   "$CALIBRATION_PROMPTS_B" "$CALIBRATION_PROMPTS_A"; do', body)
        self.assertLess(body.index('done < "$prompts"'),
                        body.index("grep -q '^RHP16_CALIBRATION_JSON '"))
        self.assertIn('grep -q \'^RHP16_CALIBRATION_JSON \' "$helper_log" && break', body)


if __name__ == "__main__":
    unittest.main()
