from __future__ import annotations

import contextlib
import json
import os
import tempfile
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from bigcherry.jobs import monitor, runner


class MonitorPolicyTests(unittest.TestCase):
    def test_stall_detector_treats_cpu_or_output_growth_as_progress(self):
        detector = monitor.StallDetector(10.0)
        self.assertFalse(detector.observe(monitor.ProgressSample(0.0, 10, 100)))
        self.assertFalse(detector.observe(monitor.ProgressSample(9.0, 11, 100)))
        self.assertFalse(detector.observe(monitor.ProgressSample(18.0, 11, 101)))
        self.assertFalse(detector.observe(monitor.ProgressSample(27.0, 12, 101)))
        self.assertTrue(detector.observe(monitor.ProgressSample(37.0, 12, 101)))

    def test_disk_guard_checks_absolute_and_fractional_floor(self):
        guard = monitor.DiskGuard(
            "root", Path("/"), min_free_bytes=100, min_free_fraction=0.05
        )
        with patch.object(
            monitor.shutil,
            "disk_usage",
            return_value=SimpleNamespace(total=1000, used=960, free=40),
        ):
            with self.assertRaises(monitor.MonitorError) as ctx:
                monitor.check_disk_guards((guard,))
        self.assertIn("free_bytes=40 < 100", str(ctx.exception))
        self.assertIn("free_fraction=0.0400 < 0.0500", str(ctx.exception))

    @unittest.skipUnless(sys.platform.startswith("linux"), "disk guards resolve POSIX roots on the Linux campaign host")
    def test_environment_disk_policy_guards_system_work_and_temp(self):
        with patch.object(monitor.tempfile, "gettempdir", return_value="/tmp-for-test"):
            guards = monitor.disk_guards_from_environment(
                project_root=Path("/project"),
                work_root=Path("/work"),
                environment={},
            )
        self.assertEqual([guard.name for guard in guards], ["system-root", "work-root", "temp-root"])
        self.assertEqual(guards[0].path, Path(os.path.abspath(os.sep)))
        self.assertEqual(guards[0].min_free_bytes, 0)
        self.assertEqual(guards[0].min_free_fraction, 0.05)
        self.assertEqual(guards[1].min_free_bytes, 0)
        self.assertEqual(guards[1].min_free_fraction, 0.02)
        self.assertEqual(guards[2].path, Path("/tmp-for-test"))
        self.assertEqual(guards[2].min_free_fraction, 0.05)

    def test_environment_disk_policy_absolute_overrides_and_project_guard(self):
        with patch.object(monitor.tempfile, "gettempdir", return_value="/tmp-for-test"):
            guards = monitor.disk_guards_from_environment(
                project_root=Path("/project"),
                work_root=Path("/work"),
                environment={
                    "BIGCHERRY_ROOT_MIN_FREE_GIB": "50",
                    "BIGCHERRY_WORK_MIN_FREE_GIB": "200",
                    "BIGCHERRY_TMP_MIN_FREE_GIB": "20",
                    "BIGCHERRY_PROJECT_MIN_FREE_GIB": "1.5",
                    "BIGCHERRY_ROOT_MIN_FREE_FRACTION": "0",
                    "BIGCHERRY_WORK_MIN_FREE_FRACTION": "0.1",
                    "BIGCHERRY_TMP_MIN_FREE_FRACTION": "0",
                    "BIGCHERRY_PROJECT_MIN_FREE_FRACTION": "0.03",
                },
            )
        self.assertEqual([guard.name for guard in guards], ["system-root", "work-root", "temp-root", "project-root"])
        self.assertEqual(guards[0].min_free_bytes, 50 * 1024**3)
        self.assertEqual(guards[1].min_free_bytes, 200 * 1024**3)
        self.assertEqual(guards[1].min_free_fraction, 0.1)
        self.assertEqual(guards[2].min_free_bytes, 20 * 1024**3)
        self.assertEqual(guards[3].min_free_bytes, int(1.5 * 1024**3))
        self.assertEqual(guards[3].min_free_fraction, 0.03)

    def test_pre_spawn_disk_pressure_returns_retryable_without_launch(self):
        policy = monitor.MonitorPolicy(poll_seconds=0.01)
        with patch.object(
            monitor, "check_disk_guards", side_effect=monitor.MonitorError("full")
        ), patch.object(monitor.subprocess, "Popen") as popen:
            result = monitor.run_monitored(
                ("fake",),
                environment={},
                guards=(),
                progress_paths=(),
                policy=policy,
            )
        self.assertEqual(result.returncode, 75)
        self.assertEqual(result.incident, "disk_pressure")
        popen.assert_not_called()

    def test_launch_failure_is_harness_error(self):
        policy = monitor.MonitorPolicy(poll_seconds=0.01)
        with patch.object(monitor, "check_disk_guards"), patch.object(
            monitor.subprocess, "Popen", side_effect=OSError("boom")
        ):
            result = monitor.run_monitored(
                ("fake",),
                environment={},
                guards=(),
                progress_paths=(),
                policy=policy,
            )
        self.assertEqual(result.returncode, 76)
        self.assertEqual(result.incident, "launch_error")


class RunnerMonitorIntegrationTests(unittest.TestCase):
    def test_runner_persists_monitor_incident_and_retry_class(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            attempt_root = root / "attempt"
            attempt_root.mkdir()
            shared_root = root / "shared"
            shared_root.mkdir()
            project_root = root / "project"
            (project_root / "tools").mkdir(parents=True)
            attempt = {
                "execution_id": "exec-1",
                "shared_root": str(shared_root),
                "project_root": str(project_root),
                "job": {"architecture": "gfx1100", "model": "/model.gguf"},
            }
            (attempt_root / "attempt.json").write_text(json.dumps(attempt))

            with patch.object(
                runner,
                "_runtime_gpu_preflight",
                return_value=((0,), {"stable_gpu_ids": ["gpu-a"]}),
            ), patch.object(
                runner, "campaign_argv", return_value=("fake-campaign",)
            ), patch.object(
                runner, "disk_guards_from_environment", return_value=()
            ), patch.object(
                runner.MonitorPolicy, "from_environment", return_value=monitor.MonitorPolicy()
            ), patch.object(
                runner,
                "run_monitored",
                return_value=monitor.MonitorResult(75, "disk_pressure", "root full"),
            ), patch.object(
                runner, "_device_locks", return_value=contextlib.nullcontext()
            ):
                rc = runner.run_attempt(attempt_root)

            self.assertEqual(rc, 75)
            result = json.loads(
                (attempt_root / "executor-result.json").read_text(encoding="utf-8")
            )
            self.assertEqual(result["failure_class"], "transient_environment")
            self.assertEqual(result["incident"], "disk_pressure")
            self.assertEqual(result["error"], "root full")
            self.assertTrue((attempt_root / "allocation-attestation.json").is_file())


if __name__ == "__main__":
    unittest.main()
