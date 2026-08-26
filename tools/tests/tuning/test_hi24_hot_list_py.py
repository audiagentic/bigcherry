"""HI24 steps 5-6: tests for inventory.write_hot_list()/_native_medians().

The impact ranking (calls x est_bytes, upgraded to calls x native_median_us
once a prior tune's measurements are available) is what GGML_HIP_TUNE_HOT_
SIGNATURES loads to decide which signatures skip screening's noise-driven
elimination -- see test_hi24_hot_list.py for the C++-side source-contract
half of this feature.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry import inventory as inv  # noqa: E402

SIG_A = "a" * 32  # 13,440 calls, small est_bytes/native median -- low impact
SIG_B = "b" * 32  # 11,960 calls, large est_bytes/native median -- high impact


def _record(observations):
    return inv.Record(header={"kind": "header"}, observations=observations)


class WriteHotListPass1Tests(unittest.TestCase):
    """calls x est_bytes, when no prior measurements are supplied."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.output = Path(self.tmp.name) / "record.hot"

    def test_impact_beats_call_count(self):
        # SIG_A has more calls but far smaller est_bytes; SIG_B must rank
        # first despite fewer calls. A ranking that reproduces call order
        # has silently fallen back to counting, which is the failure this
        # item exists to fix.
        record = _record(
            [
                {"signature": SIG_A, "calls": 13440, "est_bytes": 1000},
                {"signature": SIG_B, "calls": 11960, "est_bytes": 500000},
            ]
        )
        summary = inv.write_hot_list(record, self.output)
        self.assertEqual(summary["basis"], "calls_x_est_bytes")
        self.assertEqual(summary["rows"][0]["signature"], SIG_B)
        self.assertEqual(summary["rows"][0]["rank"], 1)

    def test_output_is_byte_stable_across_runs(self):
        record = _record(
            [
                {"signature": SIG_A, "calls": 100, "est_bytes": 100},
                {"signature": SIG_B, "calls": 100, "est_bytes": 100},
            ]
        )
        inv.write_hot_list(record, self.output)
        first = self.output.read_text(encoding="utf-8")
        inv.write_hot_list(record, self.output)
        second = self.output.read_text(encoding="utf-8")
        self.assertEqual(first, second)
        # Equal impact ties break on the signature digest, not insertion
        # order -- a < b, so SIG_A must lead when their impacts are equal.
        data_lines = [line for line in first.splitlines() if not line.startswith("#")]
        self.assertTrue(data_lines[0].startswith(SIG_A))

    def test_header_states_the_basis_used(self):
        record = _record([{"signature": SIG_A, "calls": 1, "est_bytes": 1}])
        inv.write_hot_list(record, self.output)
        text = self.output.read_text(encoding="utf-8")
        self.assertIn("# basis calls_x_est_bytes", text)

    def test_observation_without_a_signature_is_skipped(self):
        record = _record(
            [
                {"calls": 999, "est_bytes": 999},  # malformed: no signature
                {"signature": SIG_A, "calls": 1, "est_bytes": 1},
            ]
        )
        summary = inv.write_hot_list(record, self.output)
        self.assertEqual(summary["signatures"], 1)


class WriteHotListPass2Tests(unittest.TestCase):
    """calls x native_median_us, once a prior measurements file exists."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.output = Path(self.tmp.name) / "record.hot"
        self.measurements = Path(self.tmp.name) / "tune.measurements.jsonl"

    def _write_measurements(self, rows):
        lines = [json.dumps({"kind": "header"})]
        lines += [json.dumps(r) for r in rows]
        self.measurements.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def test_upgrades_basis_and_reweights_by_real_timing(self):
        # SIG_A: high call count, tiny native median (cheap per call).
        # SIG_B: fewer calls, huge native median (expensive per call) --
        # native_median_us must dominate the ranking once available, not
        # est_bytes.
        self._write_measurements(
            [
                {
                    "kind": "result",
                    "signature": SIG_A,
                    "native": "native",
                    "candidates": [{"name": "native", "median_us": 1.0}],
                },
                {
                    "kind": "result",
                    "signature": SIG_B,
                    "native": "native",
                    "candidates": [{"name": "native", "median_us": 1000.0}],
                },
            ]
        )
        record = _record(
            [
                {"signature": SIG_A, "calls": 13440, "est_bytes": 999999},
                {"signature": SIG_B, "calls": 11960, "est_bytes": 1},
            ]
        )
        summary = inv.write_hot_list(record, self.output, measurements=self.measurements)
        self.assertEqual(summary["basis"], "calls_x_native_median")
        self.assertEqual(summary["rows"][0]["signature"], SIG_B)

    def test_native_resolved_from_named_field_not_first_match(self):
        # A result carrying multiple *:native:v1-shaped candidates resolves
        # via the row's own "native" field, never the first candidate whose
        # name happens to match a naive pattern.
        self._write_measurements(
            [
                {
                    "kind": "result",
                    "signature": SIG_A,
                    "native": "mmq:native:v1",
                    "candidates": [
                        {"name": "blas:native:v1", "median_us": 50.0},
                        {"name": "mmq:native:v1", "median_us": 5.0},
                    ],
                }
            ]
        )
        record = _record([{"signature": SIG_A, "calls": 10, "est_bytes": 1}])
        medians = inv._native_medians(self.measurements, record)
        self.assertEqual(medians[SIG_A], 5.0)

    def test_missing_measurements_file_falls_back_to_pass_1(self):
        record = _record([{"signature": SIG_A, "calls": 1, "est_bytes": 1}])
        missing = Path(self.tmp.name) / "does-not-exist.jsonl"
        summary = inv.write_hot_list(record, self.output, measurements=missing)
        self.assertEqual(summary["basis"], "calls_x_est_bytes")


if __name__ == "__main__":
    unittest.main()
