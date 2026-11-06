"""RHP16 lab-only calibration launcher guards; no build, device or GPU required."""
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
            env = dict(os.environ, BC_WORK_ROOT=root)
            return subprocess.run(
                ["bash", str(LAUNCHER), "start", *flags],
                cwd=REPO,
                env=env,
                capture_output=True,
                text=True,
                timeout=10,
            )

    def test_calibrate_requires_explicit_lab_experiment(self) -> None:
        r = self.run_start("--calibrate", "--live-policy", "/policy.json")
        self.assertEqual(r.returncode, 2, r.stderr)
        self.assertIn("lab-only", r.stderr)
        self.assertNotIn("gpu", r.stderr.lower())

    def test_calibrate_requires_policy_override(self) -> None:
        r = self.run_start("--calibrate", "--experiment", "/experiment.json")
        self.assertEqual(r.returncode, 2, r.stderr)
        self.assertIn("--live-policy", r.stderr)

    def test_calibrate_requires_workload_prompts(self) -> None:
        r = self.run_start(
            "--calibrate", "--experiment", "/experiment.json",
            "--live-policy", "/policy.json",
        )
        self.assertEqual(r.returncode, 2, r.stderr)
        self.assertIn("--calibration-prompts", r.stderr)

    def test_calibrate_rejects_disabled_measurement_source(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w") as prompt:
            prompt.write("Example calibration workload\n")
            prompt.flush()
            r = self.run_start(
                "--calibrate", "--experiment", "/experiment.json",
                "--live-policy", "/policy.json",
                "--calibration-prompts", prompt.name, "--record-heat", "off",
            )
            self.assertEqual(r.returncode, 2, r.stderr)
            self.assertIn("histogram", r.stderr)

    def test_calibrate_requires_active_policy_hooks(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            exp, policy, prompts = [Path(root) / n for n in ("experiment.json", "policy.json", "prompts.txt")]
            exp.write_text("{}")
            policy.write_text('{"enabled": false}')
            prompts.write_text("A real calibration prompt\n")
            r = self.run_start(
                "--calibrate", "--experiment", str(exp),
                "--live-policy", str(policy), "--calibration-prompts", str(prompts),
            )
            self.assertEqual(r.returncode, 2, r.stderr)
            self.assertIn("enabled=true", r.stderr)

    def test_calibrate_requires_production_equivalent_plan(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            exp, policy, prompts = [Path(root) / n for n in ("experiment.json", "policy.json", "prompts.txt")]
            exp.write_text('{"engine": {"profile_ops": "on"}}')
            policy.write_text('{"enabled": true}')
            prompts.write_text("A real calibration prompt\n")
            r = self.run_start(
                "--calibrate", "--experiment", str(exp),
                "--live-policy", str(policy), "--calibration-prompts", str(prompts),
            )
            self.assertEqual(r.returncode, 2, r.stderr)
            self.assertIn("empty JSON object", r.stderr)

    def test_calibrate_rejects_blank_prompt_file(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w") as prompt:
            prompt.write("\n  \n")
            prompt.flush()
            r = self.run_start(
                "--calibrate", "--experiment", "/experiment.json",
                "--live-policy", "/policy.json",
                "--calibration-prompts", prompt.name,
            )
            self.assertEqual(r.returncode, 2, r.stderr)
            self.assertIn("nonblank", r.stderr)


if __name__ == "__main__":
    unittest.main()
