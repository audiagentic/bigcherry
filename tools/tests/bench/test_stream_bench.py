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
    delay = 0.01            # seconds per token
    fail = False
    seen_prompts: list = []

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).seen_prompts.append(body["messages"][0]["content"])
        if type(self).fail:
            self.send_response(500)
            self.end_headers()
            return
        n = int(body["max_tokens"])
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for i in range(n):
            time.sleep(type(self).delay)
            chunk = {"choices": [{"delta": {"content": f"t{i} "}}]}
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()
        usage = {"choices": [], "usage": {"prompt_tokens": len(body["messages"][0]["content"].split()), "completion_tokens": n}}
        self.wfile.write(f"data: {json.dumps(usage)}\n\ndata: [DONE]\n\n".encode())
        self.wfile.flush()


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
        _Handler.fail = False
        _Handler.delay = 0.01
        _Handler.seen_prompts = []

    def test_streams_run_in_parallel_and_the_books_add_up(self):
        one = sb.run_round(self.url, "m", 1, 16, 20, 0.0, 30.0, 1, False)
        four = sb.run_round(self.url, "m", 4, 16, 20, 0.0, 30.0, 1, False)
        self.assertTrue(one["ok"] and four["ok"])
        self.assertLess(four["wall_s"], one["wall_s"] * 2.5)             # not four times serial
        self.assertEqual(four["aggregate"]["tokens"], 80)
        self.assertGreater(four["aggregate"]["aggregate_tps"], one["aggregate"]["aggregate_tps"] * 2.0)
        for s in four["per_stream"]:
            self.assertEqual(s["tokens"], 20)
            self.assertTrue(s["text_sha256"])
            self.assertGreater(s["prompt_tokens"], 0)

    def test_the_same_text_hashes_the_same(self):
        a = sb.run_round(self.url, "m", 2, 8, 6, 0.0, 30.0, 1, False)
        b = sb.run_round(self.url, "m", 2, 8, 6, 0.0, 30.0, 1, False)
        self.assertEqual([s["text_sha256"] for s in a["per_stream"]], [s["text_sha256"] for s in b["per_stream"]])

    def test_prompts_differ_per_stream_unless_shared(self):
        sb.run_round(self.url, "m", 3, 32, 2, 0.0, 30.0, 1, False)
        self.assertEqual(len(set(_Handler.seen_prompts)), 3)
        _Handler.seen_prompts = []
        sb.run_round(self.url, "m", 3, 32, 2, 0.0, 30.0, 1, True)
        self.assertEqual(len(set(_Handler.seen_prompts)), 1)

    def test_a_failed_stream_fails_the_round_and_reports_no_aggregate(self):
        _Handler.fail = True
        rec = sb.run_round(self.url, "m", 2, 8, 4, 0.0, 30.0, 1, False)
        self.assertFalse(rec["ok"])
        self.assertNotIn("aggregate", rec)
        self.assertTrue(all(s["error"] for s in rec["per_stream"]))

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
            for key in ("agg_tps", "ttft_p95_ms", "itl_p95_ms", "stream_tps"):
                self.assertIn(f"s{n}_{key}", flat)
        _Handler.fail = True
        bad = sb.run(self.url, "m", [1], 4, 2, 1)
        with self.assertRaises(sb.StreamBenchError):
            sb.flat_metrics(bad)

    def test_inter_token_gaps_are_recorded(self):
        _Handler.delay = 0.02
        rec = sb.run_round(self.url, "m", 1, 8, 10, 0.0, 30.0, 1, False)
        self.assertGreater(rec["aggregate"]["itl_ms_p50"], 10.0)


if __name__ == "__main__":
    unittest.main()
