"""Unit tests for tools/verify_slice_a.py, using deliberately malformed
synthetic artifacts.

The close-out review (session ses_b48e80b4088a4219) found two verifier
predicates weaker than their labels: the per-result twin cardinality check
reduced to a tautology plus a totals comparison (which balances 2+0 as
1+1), and the MMQ J-best preference check could never fail. These tests pin
both against crafted artifacts so the predicates cannot regress silently.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import verify_slice_a as vsa  # noqa: E402


def header(flag):
    return {"kind": "header", "double_native": flag}


def cand(name, status="ok"):
    return {"name": name, "status": status, "median_us": 10.0}


def result(
    dispatch,
    native,
    reason="native retained",
    measured=3,
    candidates=None,
    pair="",
    canary_pct=1.5,
    winner=None,
):
    r = {
        "kind": "result",
        "dispatch": dispatch,
        "native": native,
        "winner": winner or native,
        "provisional_winner": winner or native,
        "reason": reason,
        "generated": 100,
        "applicable": 50,
        "eligible": measured,
        "measured": measured,
        "schedule": [],
        "canary_pair": pair,
    }
    if canary_pct is not None:
        r["canary_pct"] = canary_pct
    if candidates is not None:
        r["candidates"] = candidates
    return r


def on_ok_artifact():
    """A minimal valid ON artifact: one non-MMQ twin row, one MMQ J-best
    row, one MMQ twin-fallback row."""
    h = header(1)
    results = [
        result(
            "d0001",
            "mmvf:native:v1",
            candidates=[
                cand("mmvf:native:v1"),
                cand("mmvf:f16:w4:nw8:v1"),
                cand("mmvf:native:v1#twin"),
            ],
            pair="mmvf:native:v1#twin",
        ),
        result(
            "d0002",
            "mmq:native:v1",
            candidates=[
                cand("mmq:native:v1"),
                cand("mmq:q4_0:j16:fb1:v1"),
                cand("mmq:native:v1#twin"),
            ],
            pair="mmq:q4_0:j16:fb1:v1",
        ),
        result(
            "d0003",
            "mmq:native:v1",
            candidates=[cand("mmq:native:v1"), cand("mmq:native:v1#twin")],
            pair="mmq:native:v1#twin",
        ),
    ]
    return h, results, 0


class VerifierPredicateTests(unittest.TestCase):
    def test_valid_on_artifact_passes(self):
        h, results, bad = on_ok_artifact()
        failures, report = vsa.evaluate(h, results, bad, "on")
        self.assertEqual(failures, [])
        self.assertEqual(report, {"jbest": 1, "twin_fallback": 2})

    def test_valid_off_artifact_passes(self):
        h = header(0)
        results = [
            result(
                "d0001",
                "mmvf:native:v1",
                candidates=[cand("mmvf:native:v1"), cand("mmvf:f16:w4:nw8:v1")],
            ),
            result(
                "d0002",
                "mmq:native:v1",
                candidates=[cand("mmq:native:v1"), cand("mmq:q4_0:j16:fb1:v1")],
                pair="mmq:q4_0:j16:fb1:v1",
            ),
        ]
        failures, _ = vsa.evaluate(h, results, 0, "off")
        self.assertEqual(failures, [])

    def test_off_nonmmq_completed_row_with_pair_fails(self):
        h = header(0)
        # Pre-Slice A, only the MMQ J-best scan emitted a canary pair; a
        # non-MMQ row with one is not old semantics.
        results = [
            result(
                "d0001",
                "mmvf:native:v1",
                candidates=[cand("mmvf:native:v1")],
                pair="mmvf:f16:w4:nw8:v1",
            )
        ]
        failures, _ = vsa.evaluate(h, results, 0, "off")
        self.assertTrue(any("pre-Slice A" in f for f in failures), failures)

    def test_off_mmq_ungrounded_pair_fails(self):
        h = header(0)
        results = [
            result(
                "d0001",
                "mmq:native:v1",
                candidates=[cand("mmq:native:v1")],
                pair="mmq:q4_0:j16:fb1:v1",
            )
        ]
        failures, _ = vsa.evaluate(h, results, 0, "off")
        self.assertTrue(any("not grounded" in f for f in failures), failures)

    # --- twin cardinality (the old tautology) -------------------------

    def test_two_twins_and_zero_twins_do_not_balance(self):
        h, results, bad = on_ok_artifact()
        # d0001 gets a duplicate twin; d0003 loses its twin. Totals would
        # still equal the number of measured results: 2 + 0 + 1 = 3.
        results[0]["candidates"].append(cand("mmvf:native:v1#twin", "ok"))
        results[2]["candidates"] = [cand("mmq:native:v1")]
        failures, _ = vsa.evaluate(h, results, bad, "on")
        self.assertTrue(
            any("d0001" in f and "2 twin row(s)" in f for f in failures), failures
        )
        self.assertTrue(
            any("d0003" in f and "0 twin row(s)" in f for f in failures), failures
        )

    def test_measured_result_without_twin_fails(self):
        h, results, bad = on_ok_artifact()
        results[1]["candidates"] = [cand("mmq:native:v1"), cand("mmq:q4_0:j16:fb1:v1")]
        failures, _ = vsa.evaluate(h, results, bad, "on")
        self.assertTrue(
            any("d0002" in f and "expected exactly 1" in f for f in failures), failures
        )

    def test_unmeasured_result_with_twin_fails(self):
        h, results, bad = on_ok_artifact()
        stub = result(
            "d0099",
            "mmvf:native:v1",
            reason="tuning disabled after fatal measurement failure",
            measured=0,
            candidates=[cand("mmvf:native:v1#twin")],
        )
        results.append(stub)
        failures, _ = vsa.evaluate(h, results, bad, "on")
        self.assertTrue(
            any("d0099" in f and "expected 0" in f for f in failures), failures
        )

    # --- MMQ canary grounding (the old never-fails check) -------------

    def test_mmq_measured_challenger_with_twin_pair_is_legitimate(self):
        h, results, bad = on_ok_artifact()
        # A measured challenger does not force the J-best pair: eligibility
        # also requires variant.primary == j_best per the C++ policy scan,
        # which the artifact does not record. The twin fallback is valid.
        results[1]["canary_pair"] = "mmq:native:v1#twin"
        failures, report = vsa.evaluate(h, results, bad, "on")
        self.assertEqual(failures, [])
        self.assertEqual(report, {"jbest": 0, "twin_fallback": 3})

    def test_mmq_ungrounded_pair_fails(self):
        h, results, bad = on_ok_artifact()
        # The pair names a candidate with no measured (ok) row in this
        # result: fabricated or stale canary.
        results[2]["canary_pair"] = "mmq:q4_0:j16:fb1:v1"
        failures, _ = vsa.evaluate(h, results, bad, "on")
        self.assertTrue(
            any("d0003" in f and "not grounded" in f for f in failures), failures
        )

    def test_mmq_rejected_challenger_pair_fails(self):
        h, results, bad = on_ok_artifact()
        # Challenger row exists but was rejected: the pair must not claim it.
        results[1]["candidates"] = [
            cand("mmq:native:v1"),
            cand("mmq:q4_0:j16:fb1:v1", "noisy"),
            cand("mmq:native:v1#twin"),
        ]
        failures, _ = vsa.evaluate(h, results, bad, "on")
        self.assertTrue(
            any("d0002" in f and "not grounded" in f for f in failures), failures
        )

    def test_mmq_twin_pair_without_ok_twin_fails(self):
        h, results, bad = on_ok_artifact()
        # Fallback pair emitted but the twin measurement was rejected.
        results[2]["candidates"] = [
            cand("mmq:native:v1"),
            cand("mmq:native:v1#twin", "noisy"),
        ]
        failures, _ = vsa.evaluate(h, results, bad, "on")
        self.assertTrue(
            any("d0003" in f and "no ok twin measurement" in f for f in failures),
            failures,
        )

    # --- non-MMQ canary requirement ------------------------------------

    def test_nonmmq_completed_row_requires_twin_pair(self):
        h, results, bad = on_ok_artifact()
        results[0]["canary_pair"] = ""
        del results[0]["canary_pct"]
        failures, _ = vsa.evaluate(h, results, bad, "on")
        self.assertTrue(
            any("d0001" in f and "is not the native twin" in f for f in failures),
            failures,
        )

    def test_nonmmq_completed_row_requires_ok_twin(self):
        h, results, bad = on_ok_artifact()
        # Twin row exists but was rejected: canary cannot claim a measurement.
        results[0]["candidates"] = [
            cand("mmvf:native:v1"),
            cand("mmvf:f16:w4:nw8:v1"),
            cand("mmvf:native:v1#twin", "noisy"),
        ]
        failures, _ = vsa.evaluate(h, results, bad, "on")
        self.assertTrue(
            any("d0001" in f and "no ok twin measurement" in f for f in failures),
            failures,
        )

    # --- promotion rejection is NOT a measurement failure --------------

    def test_fresh_confirmation_rejected_row_is_completed(self):
        h, results, bad = on_ok_artifact()
        r = result(
            "d0100",
            "blas:native:v1",
            reason="native retained (fresh confirmation rejected provisional winner)",
            candidates=[cand("blas:native:v1"), cand("blas:native:v1#twin")],
            pair="blas:native:v1#twin",
        )
        results.append(r)
        failures, _ = vsa.evaluate(h, results, bad, "on")
        self.assertFalse(any("d0100" in f for f in failures), failures)

    # --- OFF escape hatch ----------------------------------------------

    def test_off_artifact_with_twin_row_fails(self):
        h = header(0)
        results = [
            result(
                "d0001",
                "mmvf:native:v1",
                candidates=[cand("mmvf:native:v1"), cand("mmvf:native:v1#twin")],
                pair="mmvf:native:v1#twin",
            )
        ]
        failures, _ = vsa.evaluate(h, results, 0, "off")
        self.assertTrue(any("twin row(s)" in f for f in failures), failures)
        self.assertTrue(any("canary pair" in f for f in failures), failures)

    def test_off_header_flag_mismatch_fails(self):
        h = header(1)
        failures, _ = vsa.evaluate(h, [], 0, "off")
        self.assertTrue(any("double_native" in f for f in failures), failures)

    # --- artifact integrity ---------------------------------------------

    def test_unparsable_rows_fail(self):
        failures, _ = vsa.evaluate(header(1), [], 3, "on")
        self.assertTrue(any("unparsable" in f for f in failures), failures)

    def test_funnel_overflow_fails(self):
        h, results, bad = on_ok_artifact()
        results[0]["measured"] = 99  # exceeds eligible
        failures, _ = vsa.evaluate(h, results, bad, "on")
        self.assertTrue(any("funnel" in f for f in failures), failures)

    def test_duplicate_schedule_names_fail(self):
        h, results, bad = on_ok_artifact()
        results[0]["schedule"] = ["mmvf:native:v1", "mmvf:native:v1"]
        failures, _ = vsa.evaluate(h, results, bad, "on")
        self.assertTrue(any("duplicate names" in f for f in failures), failures)

    def test_twin_winner_fails(self):
        h, results, bad = on_ok_artifact()
        results[0]["winner"] = "mmvf:native:v1#twin"
        failures, _ = vsa.evaluate(h, results, bad, "on")
        self.assertTrue(any("is a twin" in f for f in failures), failures)


if __name__ == "__main__":
    unittest.main()
