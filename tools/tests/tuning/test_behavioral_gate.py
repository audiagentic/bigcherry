"""HI143: pre-promotion behavioral regression gate -- offline tests for the
pure comparison logic (compare_traces/run_gate), plus a real-HTTP round
trip for run_vector against a fake server, matching test_server_runner.py's
own fake-server pattern. No GPU required.
"""

from __future__ import annotations

import http.server
import json
import sys
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.tuning import behavioral_gate as bg  # noqa: E402
from bigcherry.tuning.server_runner import ServerRunner  # noqa: E402


def _trace(ids, draft_n=0, draft_n_accepted=0):
    return bg.BehavioralTrace(
        generated_token_ids=tuple(ids), draft_n=draft_n, draft_n_accepted=draft_n_accepted,
    )


class CompareTracesTests(unittest.TestCase):
    def test_identical_output_and_draft_stats_is_exact_pass(self):
        native = _trace([1, 2, 3], draft_n=10, draft_n_accepted=8)
        candidate = _trace([1, 2, 3], draft_n=10, draft_n_accepted=8)
        verdict = bg.compare_traces("v", native, candidate)
        self.assertEqual(verdict.verdict, "exact_pass")
        self.assertIsNone(verdict.first_output_divergence)

    def test_different_output_is_hard_fail_regardless_of_draft_stats(self):
        # HI143's explicit contract: a generated-output change is NEVER
        # acceptable, even if the candidate's draft stats look identical
        # or better -- model-visible semantics changed.
        native = _trace([1, 2, 3], draft_n=10, draft_n_accepted=8)
        candidate = _trace([1, 2, 4], draft_n=10, draft_n_accepted=8)
        verdict = bg.compare_traces("v", native, candidate)
        self.assertEqual(verdict.verdict, "hard_fail")
        self.assertEqual(verdict.first_output_divergence, 2)

    def test_first_output_divergence_index_is_exact(self):
        native = _trace([9, 9, 1, 2, 3])
        candidate = _trace([9, 9, 1, 2, 9])
        verdict = bg.compare_traces("v", native, candidate)
        self.assertEqual(verdict.verdict, "hard_fail")
        self.assertEqual(verdict.first_output_divergence, 4)

    def test_hi141_style_regression_is_hard_fail(self):
        # This is the actual real HI141 scenario shape: same-length output
        # but diverging content once the candidate's numeric drift flips a
        # greedy choice mid-generation.
        native = _trace(list(range(128)), draft_n=137, draft_n_accepted=128)
        bad = list(range(128))
        bad[40] = 9999  # a single flipped token, mid-sequence
        candidate = _trace(bad, draft_n=200, draft_n_accepted=95)
        verdict = bg.compare_traces("hi141-repro", native, candidate)
        self.assertEqual(verdict.verdict, "hard_fail")
        self.assertEqual(verdict.first_output_divergence, 40)

    def test_same_output_different_draft_stats_is_behavior_changed_not_reject(self):
        # HI143's explicit rejection of a strict per-step non-decrease
        # rule: same output, different (even worse) acceptance trace is
        # NOT an automatic fail -- it needs throughput adjudication,
        # which this module deliberately does not itself perform.
        native = _trace([1, 2, 3], draft_n=10, draft_n_accepted=8)
        candidate = _trace([1, 2, 3], draft_n=10, draft_n_accepted=6)
        verdict = bg.compare_traces("v", native, candidate)
        self.assertEqual(verdict.verdict, "behavior_changed")

    def test_same_output_better_draft_stats_is_still_behavior_changed(self):
        # An improvement is also "changed", not silently folded into
        # exact_pass -- the caller decides what to do with it.
        native = _trace([1, 2, 3], draft_n=10, draft_n_accepted=6)
        candidate = _trace([1, 2, 3], draft_n=10, draft_n_accepted=9)
        verdict = bg.compare_traces("v", native, candidate)
        self.assertEqual(verdict.verdict, "behavior_changed")

    def test_empty_output_both_sides_is_exact_pass(self):
        native = _trace([], draft_n=0, draft_n_accepted=0)
        candidate = _trace([], draft_n=0, draft_n_accepted=0)
        verdict = bg.compare_traces("v", native, candidate)
        self.assertEqual(verdict.verdict, "exact_pass")

    def test_shorter_candidate_output_is_hard_fail_at_the_shared_length(self):
        native = _trace([1, 2, 3, 4])
        candidate = _trace([1, 2])
        verdict = bg.compare_traces("v", native, candidate)
        self.assertEqual(verdict.verdict, "hard_fail")
        self.assertEqual(verdict.first_output_divergence, 2)


class BehavioralGateReportTests(unittest.TestCase):
    def test_hard_fail_property_true_if_any_vector_hard_fails(self):
        report = bg.BehavioralGateReport(verdicts=[
            bg.VectorVerdict("a", "exact_pass", _trace([1]), _trace([1])),
            bg.VectorVerdict("b", "hard_fail", _trace([1]), _trace([2]), first_output_divergence=0),
        ])
        self.assertTrue(report.hard_fail)
        self.assertFalse(report.needs_throughput_adjudication)

    def test_needs_throughput_adjudication_true_if_any_vector_behavior_changed(self):
        report = bg.BehavioralGateReport(verdicts=[
            bg.VectorVerdict("a", "exact_pass", _trace([1]), _trace([1])),
            bg.VectorVerdict("b", "behavior_changed", _trace([1], 5, 4), _trace([1], 5, 3)),
        ])
        self.assertFalse(report.hard_fail)
        self.assertTrue(report.needs_throughput_adjudication)

    def test_all_exact_pass_report(self):
        report = bg.BehavioralGateReport(verdicts=[
            bg.VectorVerdict("a", "exact_pass", _trace([1]), _trace([1])),
            bg.VectorVerdict("b", "exact_pass", _trace([2]), _trace([2])),
        ])
        self.assertFalse(report.hard_fail)
        self.assertFalse(report.needs_throughput_adjudication)

    def test_summary_is_json_serializable_and_complete(self):
        report = bg.BehavioralGateReport(verdicts=[
            bg.VectorVerdict("a", "hard_fail", _trace([1, 2], 3, 2), _trace([1, 9], 3, 2), first_output_divergence=1),
        ])
        summary = report.summary()
        json.dumps(summary)  # must not raise
        self.assertTrue(summary["hard_fail"])
        self.assertEqual(summary["vectors"][0]["first_output_divergence"], 1)
        self.assertEqual(summary["vectors"][0]["native_draft"], [3, 2])


class _FakeCompletionHandler(http.server.BaseHTTPRequestHandler):
    """Serves a fixed /completion response so run_vector's real HTTP round
    trip (JSON parsing, field extraction) is exercised end to end without
    a real server/GPU."""

    response_body: bytes = b"{}"

    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(self.response_body)


class RunVectorRealHttpTests(unittest.TestCase):
    def setUp(self):
        _FakeCompletionHandler.response_body = json.dumps({
            "tokens": [5, 6, 7],
            "timings": {"draft_n": 12, "draft_n_accepted": 9},
        }).encode()
        self.server = http.server.HTTPServer(("127.0.0.1", 0), _FakeCompletionHandler)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.runner = ServerRunner(binary=Path("unused"), model=Path("unused"), port=self.port)
        self.runner._proc = object()  # bypass launch(); only post_json is used

    def tearDown(self):
        self.server.shutdown()
        self.thread.join(timeout=5)

    def test_run_vector_extracts_token_ids_and_draft_stats(self):
        vector = bg.BehavioralVector(name="v", prompt="hello", n_predict=3)
        trace = bg.run_vector(self.runner, vector)
        self.assertEqual(trace.generated_token_ids, (5, 6, 7))
        self.assertEqual(trace.draft_n, 12)
        self.assertEqual(trace.draft_n_accepted, 9)

    def test_run_vector_raises_on_missing_tokens(self):
        _FakeCompletionHandler.response_body = json.dumps({"timings": {"draft_n": 1, "draft_n_accepted": 1}}).encode()
        vector = bg.BehavioralVector(name="v", prompt="hello", n_predict=3)
        with self.assertRaises(bg.BehavioralGateError):
            bg.run_vector(self.runner, vector)

    def test_run_vector_fails_closed_when_mtp_telemetry_missing_entirely(self):
        # gpt code review (2026-08-29): defaulting missing draft_n/
        # draft_n_accepted to 0 was a real fail-open bug -- a vector that
        # requires MTP but got no telemetry at all must raise, not silently
        # report draft_n=0 (which could then be wrongly compared as
        # "identical" to another equally-missing 0 on the other leg).
        _FakeCompletionHandler.response_body = json.dumps({"tokens": [1], "timings": {}}).encode()
        vector = bg.BehavioralVector(name="v", prompt="hello", n_predict=1)
        with self.assertRaises(bg.BehavioralGateError):
            bg.run_vector(self.runner, vector, require_mtp=True)

    def test_run_vector_fails_closed_when_draft_n_is_zero_but_required(self):
        _FakeCompletionHandler.response_body = json.dumps({
            "tokens": [1], "timings": {"draft_n": 0, "draft_n_accepted": 0},
        }).encode()
        vector = bg.BehavioralVector(name="v", prompt="hello", n_predict=1)
        with self.assertRaises(bg.BehavioralGateError):
            bg.run_vector(self.runner, vector, require_mtp=True)

    def test_run_vector_allows_missing_mtp_telemetry_when_not_required(self):
        # A deliberately non-MTP vector (require_mtp=False) is allowed to
        # report draft_n=0 -- this is an explicit opt-out, not the silent
        # default the earlier version had.
        _FakeCompletionHandler.response_body = json.dumps({"tokens": [1], "timings": {}}).encode()
        vector = bg.BehavioralVector(name="v", prompt="hello", n_predict=1)
        trace = bg.run_vector(self.runner, vector, require_mtp=False)
        self.assertEqual(trace.draft_n, 0)
        self.assertEqual(trace.draft_n_accepted, 0)

    def test_run_vector_raises_on_draft_n_accepted_out_of_range(self):
        _FakeCompletionHandler.response_body = json.dumps({
            "tokens": [1], "timings": {"draft_n": 3, "draft_n_accepted": 5},
        }).encode()
        vector = bg.BehavioralVector(name="v", prompt="hello", n_predict=1)
        with self.assertRaises(bg.BehavioralGateError):
            bg.run_vector(self.runner, vector)

    def test_run_vector_raises_when_timings_field_absent_entirely(self):
        _FakeCompletionHandler.response_body = json.dumps({"tokens": [1]}).encode()
        vector = bg.BehavioralVector(name="v", prompt="hello", n_predict=1)
        with self.assertRaises(bg.BehavioralGateError):
            bg.run_vector(self.runner, vector)


if __name__ == "__main__":
    unittest.main()
