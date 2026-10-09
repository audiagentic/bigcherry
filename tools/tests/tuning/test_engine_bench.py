"""MEN03: the engine-neutral bench against a fake OpenAI-compatible server process (no GPU, no real engine)."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bigcherry.tuning import engine_bench  # noqa: E402

# A server that answers the routes the bench uses. It is started the way each engine declares (its own model, host
# and port flags) and records how it was launched and stopped, so one script stands in for both engines.
_FAKE = textwrap.dedent(
    '''\
    import json, os, signal, sys, time
    from http.server import BaseHTTPRequestHandler, HTTPServer

    args = sys.argv[1:]
    port = int(args[args.index("--port") + 1])
    model = args[args.index("--model") + 1] if "--model" in args else args[args.index("-m") + 1]
    print("ARGS " + json.dumps(args), flush=True)
    print("ENV " + json.dumps({k: os.environ.get(k) for k in ("LLAMA_SERVER_ENABLE_SHUTDOWN", "RADIANCE_HOME", "FAKE_X", "LEAK")}), flush=True)

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, body, kind="application/json"):
            data = body.encode()
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                self._send("{}")
            elif self.path == "/v1/models":
                self._send(json.dumps({"data": [{"id": "served-" + os.path.basename(model)}]}))
            elif self.path == "/metrics":
                self._send("# HELP x\\nradiance:draft_tokens_total 40\\nradiance:draft_accepted_total 10\\n", "text/plain")
            else:
                self.send_error(404)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/shutdown":
                self._send("{}")
                print("draft acceptance = 0.50000 (    6 accepted /    12 generated)", flush=True)
                os._exit(0)
            if "FAIL_DEPTH" in os.environ and len(body["prompt"]) > int(os.environ["FAIL_DEPTH"]):
                self.send_error(500)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            n = body["max_tokens"]
            for i in range(n):
                self.wfile.write(("data: " + json.dumps({"choices": [{"text": "t%d " % i}]}) + "\\n\\n").encode())
                self.wfile.flush()
                time.sleep(0.01)
            usage = {"prompt_tokens": len(body["prompt"]) // 4, "completion_tokens": n}
            self.wfile.write(("data: " + json.dumps({"choices": [], "usage": usage}) + "\\n\\ndata: [DONE]\\n\\n").encode())

    signal.signal(signal.SIGINT, lambda *a: os._exit(0))
    HTTPServer(("127.0.0.1", port), H).serve_forever()
    '''
)


class SummariseTests(unittest.TestCase):
    def test_rates_follow_the_stated_definitions(self):
        row = engine_bench.summarise(depth=8, rep=1, started=10.0, first=12.0, last=16.0,
                                     usage={"prompt_tokens": 1000, "completion_tokens": 41}, text="x" * 500)
        self.assertEqual((row.prompt_tokens, row.completion_tokens), (1000, 41))
        self.assertAlmostEqual(row.ttft_s, 2.0)
        self.assertAlmostEqual(row.prefill_tps, 500.0)   # prompt tokens / time to first token
        self.assertAlmostEqual(row.decode_tps, 10.0)     # (completion - 1) / (last - first)
        self.assertEqual(len(row.text_head), 160)

    def test_a_request_without_tokens_has_no_rates(self):
        row = engine_bench.summarise(depth=8, rep=0, started=1.0, first=None, last=None, usage={}, text="")
        self.assertIsNone(row.ttft_s)
        self.assertIsNone(row.prefill_tps)
        self.assertIsNone(row.decode_tps)
        one = engine_bench.summarise(depth=8, rep=0, started=1.0, first=2.0, last=2.0,
                                     usage={"prompt_tokens": 5, "completion_tokens": 1}, text="a")
        self.assertIsNone(one.decode_tps)

    def test_prompt_is_the_corpus_cut_to_depth_behind_a_nonce(self):
        prompt = engine_bench.prompt_for("abcdefghij" * 100, 10, "n1")
        self.assertTrue(prompt.startswith("[n1]\n" + ("abcdefghij" * 100)[:36]))
        self.assertTrue(prompt.endswith(engine_bench.ASK))


@unittest.skipIf(os.name == "nt", "the fake server is stopped with SIGINT for radiance; POSIX only")
class EngineBenchRunTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="engine-bench-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "bin").mkdir()
        script = self.root / "bin" / "fake_server.py"
        script.write_text(_FAKE, encoding="utf-8")
        self.binary = self.root / "bin" / "server"
        self.binary.write_text(f"#!/bin/sh\nexec {sys.executable} {script} \"$@\"\n", encoding="utf-8")
        self.binary.chmod(0o755)
        self.model = self.root / "model.bin"
        self.model.write_text("m", encoding="utf-8")
        self.corpus = self.root / "corpus.txt"
        self.corpus.write_text("lorem ipsum dolor sit amet " * 400, encoding="utf-8")

    def _run(self, engine, **kwargs):
        lines: list[str] = []
        record = engine_bench.run(
            engine=engine, binary=self.binary, model=str(self.model), out_dir=self.root / "out", depths=(50, 200),
            corpus_path=self.corpus, reps=2, decode=4, health_timeout_s=30, report=lines.append, **kwargs,
        )
        return record, lines

    def test_both_engines_write_the_same_record_shape(self):
        os.environ["LEAK"] = "1"
        self.addCleanup(os.environ.pop, "LEAK", None)
        llama, _ = self._run("llamacpp", label="llama", env={"FAKE_X": "1"}, env_unset=("LEAK",),
                             extra_args=("-c", "4096"))
        radiance, lines = self._run("radiance", label="rad", extra_args=("--tp", "1"))

        self.assertEqual(sorted(llama), sorted(radiance))
        for record in (llama, radiance):
            self.assertEqual(record["schema"], engine_bench.SCHEMA)
            self.assertIsNone(record["server_error"])
            self.assertEqual(record["served_model"], "served-model.bin")
            self.assertEqual([(r["depth"], r["rep"]) for r in record["rows"]], [(50, 0), (50, 1), (200, 0), (200, 1)])
            self.assertTrue(all(r["completion_tokens"] == 4 and r["prefill_tps"] and r["decode_tps"] and not r["error"]
                                for r in record["rows"]))
            self.assertTrue(record["shutdown"]["clean"])
            saved = json.loads((self.root / "out" / f"{record['label']}.{engine_bench.RESULT}").read_text())
            self.assertEqual(saved, record)

        # each engine was started and stopped the way its declaration says, and its drafter totals were found there
        llama_log = (self.root / "out" / "llama.server.log").read_text()
        rad_log = (self.root / "out" / "rad.server.log").read_text()
        self.assertIn('"-m"', llama_log)
        self.assertIn('"-c", "4096"', llama_log)
        self.assertIn('"LLAMA_SERVER_ENABLE_SHUTDOWN": "1"', llama_log)
        self.assertIn('"FAKE_X": "1"', llama_log)
        self.assertIn('"LEAK": null', llama_log)
        self.assertEqual(llama["shutdown"]["method"], "http")
        self.assertEqual(llama["draft"], {"drafted": 12, "accepted": 6})
        self.assertIn('"--model"', rad_log)
        self.assertIn("radiance_home", rad_log)
        self.assertEqual(radiance["shutdown"]["method"], "sigint")
        self.assertEqual(radiance["draft"], {"drafted": 40, "accepted": 10})
        self.assertIn("draft: accepted 10 of 40 (25.0%)", lines)

    def test_a_failed_depth_is_recorded_and_the_server_is_still_stopped(self):
        record, lines = self._run("llamacpp", label="fail", env={"FAIL_DEPTH": "400"})
        self.assertEqual([(r["depth"], r["rep"], bool(r["error"])) for r in record["rows"]],
                         [(50, 0, False), (50, 1, False), (200, 0, True)])
        self.assertTrue(any("REQUEST_FAILED" in line for line in lines))
        self.assertTrue(record["shutdown"]["clean"])

    def test_a_server_that_never_starts_is_reported_not_raised(self):
        self.binary.write_text("#!/bin/sh\necho no such kernel library\nexit 3\n", encoding="utf-8")
        record, lines = self._run("radiance", label="dead")
        self.assertIn("no such kernel library", record["server_error"])
        self.assertEqual(record["rows"], [])
        self.assertTrue(any(line.startswith("SERVER_FAILED") for line in lines))
        self.assertTrue((self.root / "out" / f"dead.{engine_bench.RESULT}").is_file())


if __name__ == "__main__":
    unittest.main()
