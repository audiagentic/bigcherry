"""HI16: offline tests for hi16_forced_native_parity.py's parsing/assertion
logic -- check_result_row()/check_measurements_file()/parity_rows(). No
real binary or hardware needed for this half; run_case()'s actual
subprocess invocation is exercised on real hardware separately (see
HI16.md notes for the real-GPU results this script was validated against).
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bigcherry import hi16_forced_native_parity as parity  # noqa: E402


def _row(*, dispatch="aa", canary_pair="", candidates):
    return {"kind": "result", "dispatch": dispatch, "canary_pair": canary_pair, "candidates": candidates}


class ParityRowsTests(unittest.TestCase):
    def test_finds_the_twin(self):
        candidates = [
            {"name": "native", "nmse": 0, "max_abs": 0, "status": "ok"},
            {"name": "native#twin", "nmse": 0, "max_abs": 0, "status": "ok"},
        ]
        rows = parity.parity_rows(candidates, canary_pair="")
        self.assertEqual([r["name"] for r in rows], ["native#twin"])

    def test_finds_the_named_j_best_pair_when_not_the_twin(self):
        candidates = [
            {"name": "native", "nmse": 0, "max_abs": 0, "status": "ok"},
            {"name": "mmq:q4_0:j16:v1", "nmse": 0, "max_abs": 0, "status": "ok"},
            {"name": "mmq:q4_0:j32:v1", "nmse": 0, "max_abs": 0, "status": "ok"},
        ]
        rows = parity.parity_rows(candidates, canary_pair="mmq:q4_0:j16:v1")
        self.assertEqual([r["name"] for r in rows], ["mmq:q4_0:j16:v1"])

    def test_ordinary_challengers_are_out_of_scope(self):
        candidates = [{"name": "mmvq:j64:v1", "nmse": 0.02, "max_abs": 0.5, "status": "ok"}]
        rows = parity.parity_rows(candidates, canary_pair="")
        self.assertEqual(rows, [])


class CheckResultRowTests(unittest.TestCase):
    def test_exact_zero_passes(self):
        row = _row(candidates=[{"name": "native#twin", "nmse": 0, "max_abs": 0, "status": "ok"}])
        self.assertEqual(parity.check_result_row(row), [])

    def test_nonzero_nmse_fails(self):
        row = _row(candidates=[{"name": "native#twin", "nmse": 1e-9, "max_abs": 0, "status": "ok"}])
        failures = parity.check_result_row(row)
        self.assertEqual(len(failures), 1)
        self.assertIn("nmse=1e-09", failures[0])

    def test_nonzero_max_abs_fails(self):
        row = _row(candidates=[{"name": "native#twin", "nmse": 0, "max_abs": 1e-6, "status": "ok"}])
        failures = parity.check_result_row(row)
        self.assertEqual(len(failures), 1)

    def test_non_ok_status_fails(self):
        row = _row(candidates=[{"name": "native#twin", "nmse": 0, "max_abs": 0, "status": "tolerance"}])
        failures = parity.check_result_row(row)
        self.assertEqual(len(failures), 1)
        self.assertIn("status='tolerance'", failures[0])

    def test_no_parity_candidate_is_a_failure_not_a_silent_pass(self):
        row = _row(candidates=[{"name": "native", "nmse": 0, "max_abs": 0, "status": "ok"}])
        failures = parity.check_result_row(row)
        self.assertEqual(len(failures), 1)
        self.assertIn("nothing to check parity against", failures[0])


class CheckMeasurementsFileTests(unittest.TestCase):
    def test_aggregates_failures_across_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.jsonl"
            lines = [
                json.dumps({"kind": "header"}),
                json.dumps(_row(dispatch="aa", candidates=[
                    {"name": "native#twin", "nmse": 0, "max_abs": 0, "status": "ok"}
                ])),
                json.dumps(_row(dispatch="bb", candidates=[
                    {"name": "native#twin", "nmse": 0.01, "max_abs": 0, "status": "ok"}
                ])),
            ]
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            failures = parity.check_measurements_file(path)
            self.assertEqual(len(failures), 1)
            self.assertIn("dispatch bb", failures[0])


if __name__ == "__main__":
    unittest.main()
