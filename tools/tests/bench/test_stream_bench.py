"""RAD10-01: the concurrent-stream benchmark, against an in-process fake OpenAI-style streaming server (no GPU, no network)."""

from __future__ import annotations

import json
import sys
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.bench import stream_bench as sb  # noqa: E402


class _Handler(BaseHTTPRequestHandler):
    delay = 0.01            # seconds per event
    fail = False
    reasoning = False       # tokens arrive as reasoning_content
    tokens_per_event = 1    # >1 packs several tokens into one SSE event (MTP-style)
    usage = True            # send a usage block
    truncate = False        # close the stream without [DONE] or a finish reason
    malformed = False       # send an unparsable data line
    seen_prompts: list = []

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        cls = type(self)
        cls.seen_prompts.append(body["messages"][0]["content"])
        if cls.fail:
            self.send_response(500)
            self.end_headers()
            return
        n = int(body["max_tokens"])
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        if cls.malformed:
            self.wfile.write(b"data: {not json\n\n")
            self.wfile.flush()
            return
        sent = 0
        while sent < n:
            time.sleep(cls.delay)
            take = min(cls.tokens_per_event, n - sent)
            field = "reasoning_content" if cls.reasoning else "content"
            chunk = {"choices": [{"delta": {field: "".join(f"t{sent + k} " for k in range(take))}}]}
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()
            sent += take
        if cls.truncate:
            return
        self.wfile.write(f"data: {json.dumps({'choices': [{'delta': {}, 'finish_reason': 'length'}]})}\n\n".encode())
        if cls.usage:
            usage = {"choices": [], "usage": {"prompt_tokens": len(body["messages"][0]["content"].split()),
                                              "completion_tokens": n}}
            self.wfile.write(f"data: {json.dumps(usage)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def _reset():
    _Handler.fail = _Handler.reasoning = _Handler.truncate = _Handler.malformed = False
    _Handler.usage = True
    _Handler.delay = 0.01
    _Handler.tokens_per_event = 1
    _Handler.seen_prompts = []


class StreamBench(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        cls.url = f"http://127.0.0.1:{cls.server.server_address[1]}"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def setUp(self):
        _reset()

    def round(self, n=1, max_tokens=6, **kw):
        return sb.run_round(self.url, "m", n, 8, max_tokens, 0.0, 30.0, 1, False, **kw)

    def test_streams_run_in_parallel_and_the_books_add_up(self):
        one = sb.run_round(self.url, "m", 1, 16, 20, 0.0, 30.0, 1, False)
        four = sb.run_round(self.url, "m", 4, 16, 20, 0.0, 30.0, 1, False)
        self.assertTrue(one["ok"] and four["ok"])
        self.assertLess(four["wall_s"], one["wall_s"] * 2.5)             # not four times serial
        self.assertEqual(four["aggregate"]["tokens"], 80)
        self.assertGreater(four["aggregate"]["aggregate_tps"], one["aggregate"]["aggregate_tps"] * 2.0)
        for s in four["per_stream"]:
            self.assertEqual((s["tokens"], s["usage_source"], s["finish_reason"]), (20, "server", "length"))
            self.assertTrue(s["text_sha256"])
            self.assertGreater(s["prompt_tokens"], 0)
        self.assertLess(four["start_skew_ms"], 50.0)                     # the controller released them together

    def test_several_tokens_in_one_event_are_counted_from_usage_not_events(self):
        _Handler.tokens_per_event = 4
        rec = self.round(1, max_tokens=12)
        s = rec["per_stream"][0]
        self.assertEqual((s["tokens"], s["events"], s["usage_source"]), (12, 3, "server"))

    def test_no_usage_block_fails_unless_an_estimate_is_allowed(self):
        _Handler.usage = False
        rec = self.round(1)
        self.assertFalse(rec["ok"])
        self.assertIn("no usage block", rec["per_stream"][0]["error"])
        est = self.round(1, allow_estimate=True)
        self.assertTrue(est["ok"])
        self.assertEqual(est["per_stream"][0]["usage_source"], "event_count_estimate")
        self.assertTrue(est["aggregate"]["estimated_tokens"])

    def test_a_truncated_or_malformed_stream_fails_the_round(self):
        _Handler.truncate = True
        rec = self.round(1)
        self.assertFalse(rec["ok"])
        self.assertIn("truncated", rec["per_stream"][0]["error"])
        _reset()
        _Handler.malformed = True
        self.assertFalse(self.round(1)["ok"])

    def test_reasoning_tokens_count_as_stream_text(self):
        _Handler.reasoning = True
        rec = self.round(1)
        self.assertTrue(rec["ok"])
        self.assertEqual(rec["per_stream"][0]["tokens"], 6)

    def test_the_same_text_hashes_the_same(self):
        a = self.round(2)
        b = self.round(2)
        self.assertEqual([s["text_sha256"] for s in a["per_stream"]], [s["text_sha256"] for s in b["per_stream"]])

    def test_prompts_differ_per_stream_unless_shared(self):
        sb.run_round(self.url, "m", 3, 32, 2, 0.0, 30.0, 1, False)
        self.assertEqual(len(set(_Handler.seen_prompts)), 3)
        _Handler.seen_prompts = []
        sb.run_round(self.url, "m", 3, 32, 2, 0.0, 30.0, 1, True)
        self.assertEqual(len(set(_Handler.seen_prompts)), 1)

    def test_a_failed_stream_fails_the_round_and_reports_no_aggregate(self):
        _Handler.fail = True
        rec = self.round(2)
        self.assertFalse(rec["ok"])
        self.assertNotIn("aggregate", rec)
        self.assertTrue(all(s["error"] for s in rec["per_stream"]))

    def test_aggregate_is_finite_batch_makespan_and_the_tail_is_reported(self):
        rec = self.round(2, max_tokens=10)
        agg = rec["aggregate"]
        self.assertAlmostEqual(agg["aggregate_tps"], agg["tokens"] / rec["wall_s"], delta=agg["aggregate_tps"] * 0.01)
        self.assertGreaterEqual(agg["ttft_ms_max"], agg["ttft_ms_p50"])
        self.assertIsNotNone(rec["per_stream"][0]["response_tail_ms"])
        self.assertIn("finite-batch", sb.run(self.url, "m", [1], 4, 2, 1)["config"]["throughput_definition"])

    def test_run_records_config_and_summary_and_cli_exit_codes(self):
        record = sb.run(self.url, "m", [1, 2], 8, 6, 2)
        self.assertEqual(record["schema"], sb.SCHEMA)
        self.assertTrue(record["ok"])
        self.assertEqual(sorted(record["summary"]), ["1", "2"])
        self.assertEqual(len(record["rounds"]), 4)
        out = Path(__file__).with_name("_stream_bench_out.json")
        try:
            self.assertEqual(sb.main(["--url", self.url, "--streams", "1,2", "--reps", "1", "--max-tokens", "4",
                                      "--prompt-tokens", "8", "--output", str(out)]), 0)
            self.assertEqual(json.loads(out.read_text())["schema"], sb.SCHEMA)
        finally:
            out.unlink(missing_ok=True)
        _Handler.fail = True
        self.assertEqual(sb.main(["--url", self.url, "--streams", "1", "--reps", "1", "--max-tokens", "2",
                                  "--prompt-tokens", "4", "--output", str(out)]), 1)
        out.unlink(missing_ok=True)
        self.assertEqual(sb.main(["--url", self.url, "--streams", "0"]), 2)

    def test_flat_metrics_feed_the_ab_harness(self):
        record = sb.run(self.url, "m", [1, 2], 8, 6, 2)
        flat = sb.flat_metrics(record)
        for n in (1, 2):
            for key in ("agg_tps", "ttft_max_ms", "gap_p95_ms", "stream_tps"):
                self.assertIn(f"s{n}_{key}", flat)
        _Handler.fail = True
        bad = sb.run(self.url, "m", [1], 4, 2, 1)
        with self.assertRaises(sb.StreamBenchError):
            sb.flat_metrics(bad)

    def test_event_gaps_are_recorded(self):
        _Handler.delay = 0.02
        rec = self.round(1, max_tokens=10)
        self.assertGreater(rec["aggregate"]["event_gap_ms_p50"], 10.0)


if __name__ == "__main__":
    unittest.main()
