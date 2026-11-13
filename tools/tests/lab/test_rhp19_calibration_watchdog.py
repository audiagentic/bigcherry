"""RHP19 finite calibration and once-only record card-free tests."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[3]
MODULE = ROOT / "tools/lab/radiance/calibration_watchdog.py"
spec = importlib.util.spec_from_file_location("calibration_watchdog", MODULE)
watch = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(watch)


class CalibrationWatchdogTests(unittest.TestCase):
    def policy(self):
        return {"cadence_ns": 5_000_000_000, "max_heat_age_ns": 10_000_000_000,
                "min_residency_ns": 50_000_000_000, "min_samples": 89}

    def marker(self, *, missing=()):
        measured = {key: 1 for key in watch.REQUIRED - set(missing)}
        absent = {key: "no measured event" for key in missing}
        import json
        return watch.MARKER + json.dumps({"measurements": measured, "absent": absent}) + "\n"

    def test_limits_are_from_policy_and_four_workload_phases(self):
        budget, report = watch.limits(self.policy())
        self.assertEqual(budget, 4 * (90*5 + 10 + 50))
        self.assertEqual(report, 10)
        changed = self.policy()
        changed["min_samples"] *= 2
        self.assertGreater(watch.limits(changed)[0], budget)

    def test_missing_measured_policy_input_fails_closed(self):
        for name in list(self.policy()):
            p = self.policy()
            del p[name]
            with self.assertRaisesRegex(watch.CalibrationError, name):
                watch.limits(p)
        with self.assertRaisesRegex(watch.CalibrationError, "positive"):
            watch.limits({**self.policy(), "cadence_ns": 0})

    def test_real_workload_count_uses_transport_ceiling(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a", Path(tmp) / "b"
            a.write_text("a\n\na\n")
            b.write_text("b\nb\nb\n")
            bound, report, prefeed, requests = watch.workload_limits(
                self.policy(), a, b, 120)
            self.assertEqual(requests, 2 * (2 + 3) * 120)
            self.assertEqual(bound, prefeed + requests)
            self.assertEqual(prefeed, watch.limits(self.policy())[0])
            self.assertEqual(report, watch.limits(self.policy())[1])
            b.with_name("empty").write_text("")
            with self.assertRaisesRegex(watch.CalibrationError, "empty"):
                watch.workload_limits(self.policy(), a, b.with_name("empty"), 120)

    def test_measured_drift_window_includes_capacity_heat_gap_and_route_horizon(self):
        # The helper reports observed ring capacity, not policy min_samples.
        line = (watch.PROGRESS + "accepted_heat_deltas=48 drift_intervals=15 "
                "drift_capacity=16 drift_horizon_observations=336121830 "
                "observation_rate_per_s=1381865 heat_gap_max_ns=5035931069")
        seconds, ready = watch.drift_evidence(self.policy(), line)
        self.assertTrue(ready)
        self.assertEqual(seconds, 244)  # measured observation horizon is the longer wait
        increased = line.replace("heat_gap_max_ns=5035931069",
                                 "heat_gap_max_ns=30000000000")
        self.assertEqual(watch.drift_evidence(self.policy(), increased)[0], 480)
        self.assertEqual(watch.drift_evidence(self.policy(), "") , (0, False))
        self.assertEqual(watch.drift_evidence(
            self.policy(), line.replace("observation_rate_per_s=1381865",
                                        "observation_rate_per_s=0")), (0, False))

    def test_helper_progress_malformed_fails_closed(self):
        line = (watch.PROGRESS + "accepted_heat_deltas=5 drift_intervals=20 "
                "drift_capacity=16 drift_horizon_observations=12 "
                "observation_rate_per_s=2 heat_gap_max_ns=10")
        with self.assertRaisesRegex(watch.CalibrationError, "exceed capacity"):
            watch.drift_evidence(self.policy(), line)
        with self.assertRaisesRegex(watch.CalibrationError, "incomplete"):
            watch.drift_evidence(self.policy(), line.replace(" drift_capacity=16", ""))
        source = (ROOT / "engines/radiance/components/moe-remote/worker_policy_driver.h").read_text()
        self.assertIn("RHP19_CALIBRATION_PROGRESS", source)
        self.assertIn("progress.drift_capacity", source)
        self.assertIn("progress.delta_gap_ns.max()", source)

    def test_no_marker_is_waiting_not_success(self):
        self.assertIsNone(watch.record_in_log("calibration waiting: drift intervals\n"))
        self.assertIn("drift intervals", watch.waiting_reason("calibration waiting: drift intervals\n"))

    def test_missing_feed_is_named(self):
        self.assertIn("heat_watermarked", watch.waiting_reason("helper started\n"))

    def test_complete_once_only_marker(self):
        row = watch.record_in_log("helper ready\n" + self.marker())
        self.assertEqual(len(row["measurements"]), len(watch.REQUIRED))
        with self.assertRaisesRegex(watch.CalibrationError, "exactly one"):
            watch.record_in_log(self.marker() + self.marker())

    def test_malformed_marker_is_fatal(self):
        with self.assertRaisesRegex(watch.CalibrationError, "malformed"):
            watch.record_in_log(watch.MARKER + "{bad\n")

    def test_planner_partial_cannot_enter_production(self):
        with self.assertRaisesRegex(watch.CalibrationError, "tail_lane_row_ns"):
            watch.record_in_log(self.marker(missing=("tail_lane_row_ns",)))

    def test_only_effect_absence_may_be_ignored(self):
        import json
        row = json.loads(self.marker().split(" ", 1)[1])
        row["absent"] = {"abort_quiescent_ns": "no abort observed",
                         "effect_row_noise_ns": "structurally unjudgeable"}
        self.assertIsNotNone(watch.record_in_log(watch.MARKER + json.dumps(row)))

    def test_launcher_syntax_and_does_not_wait_forever(self):
        launcher = ROOT / "tools/lab/radiance/launch-flashnext-best.sh"
        body = launcher.read_text()
        self.assertIn('calibration_limits=$(python3 "$CAL_MONITOR" limits', body)
        self.assertIn('python3 "$CAL_MONITOR" marker "$helper_log"', body)
        self.assertIn('calibration_watch || exit 1', body)
        r = subprocess.run(["bash", "-n", str(launcher)],
                           capture_output=True, text=True, timeout=5)
        self.assertEqual(r.returncode, 0, r.stderr)


if __name__ == "__main__":
    unittest.main()
