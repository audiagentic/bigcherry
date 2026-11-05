"""accept-bench against an in-process fake completions server (no GPU, no network)."""

from __future__ import annotations

import hashlib
import json
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.bench import accept_bench as ab  # noqa: E402


class _Handler(BaseHTTPRequestHandler):
    drafted = 0.0
    accepted = 0.0
    steps = 0.0
    tokens = 0.0
    seconds = 0.0
    metrics_on = True
    drafts = True
    seen: list = []
    accept_rate = {"Hello": 0.5}      # prompt -> accepted fraction; everything else 1.0

    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path != "/metrics" or not self.metrics_on:
            self.send_response(404); self.end_headers(); return
        body = (f"radiance:draft_tokens_total {self.drafted}\nradiance:draft_accepted_total {self.accepted}\n"
                f"radiance:engine_steps_decode_total {self.steps}\nradiance:engine_step_seconds_decode_total {self.seconds}\n"
                f"radiance:decode_tokens_total {self.tokens}\n").encode()
        self.send_response(200); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).seen.append(req)
        n = req["max_tokens"]
        if self.drafts and n > 8:
            cls = type(self)
            cls.steps += n / 4
            cls.tokens += n
            cls.seconds += n / 4 * 0.025
            cls.drafted += n * 0.75
            cls.accepted += n * 0.75 * cls.accept_rate.get(req["prompt"], 1.0)
        body = json.dumps({"choices": [{"text": " reply to " + req["prompt"]}]}).encode()
        self.send_response(200); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)


class AcceptBenchTests(unittest.TestCase):
    def setUp(self):
        for k, v in dict(drafted=0.0, accepted=0.0, steps=0.0, tokens=0.0, seconds=0.0, metrics_on=True, drafts=True).items():
            setattr(_Handler, k, v)
        _Handler.seen = []
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def test_prompts_are_the_rad23_ones_and_requests_are_pinned(self):
        record = ab.run(self.url, max_tokens=64)
        self.assertEqual(list(record["kinds"]), ["hello", "story", "code", "list", "random"])
        pinned = _Handler.seen[1:]                                       # the first is the warm-up
        self.assertEqual([r["prompt"] for r in pinned], list(ab.KINDS.values()))
        for r in pinned:
            self.assertEqual((r["temperature"], r["max_tokens"], r["ignore_eos"]), (0.0, 64, True))
        self.assertEqual(ab.KINDS["list"], "Count from 1 to 500, one number per line.")

    def test_acceptance_comes_from_the_counter_deltas_of_each_prompt_alone(self):
        record = ab.run(self.url, kinds=("hello", "list"), max_tokens=64)
        hello, lst = record["kinds"]["hello"], record["kinds"]["list"]
        self.assertAlmostEqual(hello["acceptance"], 0.5, places=3)
        self.assertAlmostEqual(lst["acceptance"], 1.0, places=3)
        self.assertEqual((hello["drafted"], hello["accepted"]), (48, 24))  # the warm-up and the other prompt excluded
        self.assertAlmostEqual(hello["decode_tokens_per_step"], 4.0, places=3)
        self.assertAlmostEqual(hello["decode_ms_per_step"], 25.0, places=2)

    def test_the_record_carries_prompt_and_text_digests(self):
        r = ab.run(self.url, kinds=("code",), max_tokens=64)["kinds"]["code"]
        self.assertEqual(r["prompt_sha256"], hashlib.sha256(ab.KINDS["code"].encode()).hexdigest())
        self.assertEqual(r["text_sha256"], hashlib.sha256((" reply to " + ab.KINDS["code"]).encode()).hexdigest())

    def test_refusals(self):
        with self.assertRaises(ab.AcceptBenchError):
            ab.run(self.url, kinds=("nope",))
        with self.assertRaises(ab.AcceptBenchError):
            ab.run(self.url, max_tokens=0)
        _Handler.metrics_on = False
        with self.assertRaises(ab.AcceptBenchError):
            ab.run(self.url, max_tokens=64)
        _Handler.metrics_on = True
        _Handler.drafts = False
        with self.assertRaisesRegex(ab.AcceptBenchError, "drafted nothing"):
            ab.run(self.url, max_tokens=64)


if __name__ == "__main__":
    unittest.main()
