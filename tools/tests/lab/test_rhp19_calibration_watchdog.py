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

    def drift_progress(self, **changed):
        import json
        row = {"capacity": 16, "intervals": 7, "horizon_observations": 100_000_000,
               "observation_rate_per_s": 1_000_000, "accepted_gap_ns": 5_000_000_000,
               "accepted_heat_deltas": 8}
        row.update(changed)
        return watch.DRIFT_MARKER + json.dumps(row) + "\n"

    def test_helper_measured_drift_window_extends_policy_envelope(self):
        baseline, report = watch.limits(self.policy())
        measured = watch.measured_drift(self.drift_progress())
        budget, measured_report = watch.limits(self.policy(), measured)
        # 16 retained samples * max(policy cadence/age, measured gap)
        self.assertEqual(budget, baseline + 4 * 16 * 10)
        self.assertEqual(measured_report, report)
        # An observed retained horizon longer than that ring minimum wins.
        long_horizon = watch.measured_drift(
            self.drift_progress(horizon_observations=300_000_000))
        self.assertEqual(watch.limits(self.policy(), long_horizon)[0],
                         baseline + 4 * 300)
        # Larger measured gaps extend the envelope with no guessed constant.
        slow = watch.measured_drift(
            self.drift_progress(accepted_gap_ns=20_000_000_000))
        self.assertEqual(watch.limits(self.policy(), slow)[0],
                         baseline + 4 * 16 * 20)

    def test_drift_waits_for_measured_feed_and_rejects_bad_telemetry(self):
        baseline = watch.limits(self.policy())
        self.assertIsNone(watch.measured_drift("no helper progress\n"))
        for drift in ({"accepted_heat_deltas": 1}, {"accepted_gap_ns": 0}):
            progress = watch.measured_drift(self.drift_progress(**drift))
            self.assertEqual(watch.limits(self.policy(), progress), baseline)
        for log in (watch.DRIFT_MARKER + "{bad\n",
                    self.drift_progress(capacity=0),
                    self.drift_progress(intervals=17),
                    self.drift_progress(observation_rate_per_s=-1)):
            with self.assertRaises(watch.CalibrationError):
                watch.measured_drift(log)

    def test_progress_uses_latest_helper_record(self):
        progress = watch.measured_drift(
            self.drift_progress(accepted_gap_ns=2_000_000_000) +
            self.drift_progress(accepted_gap_ns=15_000_000_000))
        self.assertEqual(progress["accepted_gap_ns"], 15_000_000_000)

    def test_bounded_deadline_covers_workload_and_later_drift_growth(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / "a.txt", Path(directory) / "b.txt"
            a.write_text("first\n\nsecond\n")
            b.write_text("one\ntwo\nthree\n")
            budget = watch.workload_budget(a, b, 120)
            self.assertEqual(budget, 2 * (2 + 3) * 120)
            baseline, _ = watch.limits(self.policy())
            early = watch.measured_drift(self.drift_progress())
            later = watch.measured_drift(
                self.drift_progress(horizon_observations=300_000_000))
            self.assertEqual(watch.bounded_deadline(self.policy(), budget),
                             baseline + budget)
            first = watch.bounded_deadline(self.policy(), budget, early)
            second = watch.bounded_deadline(self.policy(), budget, later)
            self.assertGreater(first, baseline + budget)
            self.assertGreater(second, first)
            self.assertEqual(second, baseline + 2 * budget)
            extreme = watch.measured_drift(
                self.drift_progress(horizon_observations=10_000_000_000))
            self.assertEqual(
                watch.bounded_deadline(self.policy(), budget, extreme),
                baseline + 2 * budget)
            b.write_text("\n \n")
            with self.assertRaisesRegex(watch.CalibrationError, "empty"):
                watch.workload_budget(a, b, 120)

    def test_bounded_cli_emits_feed_gate_and_fixed_limit(self):
        import tempfile
        import json
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            policy, a, b, helper = [path / name for name in
                                    ("policy.json", "a.txt", "b.txt", "helper.log")]
            policy.write_text(json.dumps(self.policy()))
            a.write_text("a\n")
            b.write_text("b\n")
            helper.write_text("")
            args = [str(policy), str(a), str(b), "120", str(helper)]
            run = subprocess.run(["python3", str(MODULE), "bounded", *args],
                                 capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            total, report, prefeed, feed = map(int, run.stdout.split())
            self.assertEqual(feed, 0)
            self.assertEqual(total, prefeed + 4 * 120)
            helper.write_text(self.drift_progress())
            with_feed = subprocess.run(["python3", str(MODULE), "bounded", *args],
                                       capture_output=True, text=True)
            self.assertEqual(with_feed.returncode, 0, with_feed.stderr)
            new_total, _, _, feed = map(int, with_feed.stdout.split())
            self.assertEqual(feed, 1)
            self.assertGreater(new_total, total)

    def test_helper_republishes_drift_without_waiting_reason_change(self):
        source = (ROOT / "engines/radiance/components/moe-remote"
                  / "worker_policy_driver.h").read_text()
        self.assertIn("RHP19_DRIFT_PROGRESS_JSON ", source)
        self.assertIn('calibration_report_line("RHP19_DRIFT_PROGRESS_JSON "',
                      source)
        self.assertIn("next_window_noise_check_ns_=add_saturating", source)

    def test_launcher_syntax_and_does_not_wait_forever(self):
        launcher = ROOT / "tools/lab/radiance/launch-flashnext-best.sh"
        body = launcher.read_text()
        self.assertIn('calibration_limits=$(python3 "$CAL_MONITOR" bounded', body)
        self.assertIn('python3 "$CAL_MONITOR" marker "$helper_log"', body)
        self.assertIn('calibration_watch || exit 1', body)
        self.assertIn('calibration_have_feed=1', body)
        self.assertIn('bounded "$LIVE_POLICY_FILE"', body)
        r = subprocess.run(["bash", "-n", str(launcher)],
                           capture_output=True, text=True, timeout=5)
        self.assertEqual(r.returncode, 0, r.stderr)


if __name__ == "__main__":
    unittest.main()
