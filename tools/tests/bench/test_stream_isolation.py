"""stream_isolation: the pure parts (common prefix, flat metrics) and a full run against a stub server."""
import http.server
import json
import threading
import unittest

from bigcherry.bench import stream_isolation as si


class _Handler(http.server.BaseHTTPRequestHandler):
    corrupt_when_batched = False
    running = 0
    lock = threading.Lock()

    def log_message(self, *a):
        pass

    def do_GET(self):
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        prompt = body["messages"][0]["content"]
        reply = "answer:" + prompt[-12:]
        with _Handler.lock:
            _Handler.running += 1
            batched = _Handler.running > 1
        try:
            if _Handler.corrupt_when_batched and batched:
                reply = reply[:10] + "XX"
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for ch in (reply[:8], reply[8:]):
                self.wfile.write(("data: " + json.dumps({"choices": [{"delta": {"content": ch}}]}) + "\n\n").encode())
            self.wfile.write(("data: " + json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}],
                                                      "usage": {"prompt_tokens": 5, "completion_tokens": 4}}) + "\n\n").encode())
            self.wfile.write(b"data: [DONE]\n\n")
        finally:
            with _Handler.lock:
                _Handler.running -= 1


class StreamIsolationTests(unittest.TestCase):
    def _serve(self, corrupt):
        _Handler.corrupt_when_batched = corrupt
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.shutdown)
        return f"http://127.0.0.1:{srv.server_port}"

    def test_common_prefix(self):
        self.assertEqual(si._common_prefix("abcd", "abxd"), 2)
        self.assertEqual(si._common_prefix("abc", "abc"), 3)

    def test_clean_server_matches_everywhere(self):
        rec = si.run(self._serve(False), "m", 4, [2, 3], 8, 16)
        m = si.flat_metrics(rec)
        self.assertEqual(m["iso_solo_repeat_frac"], 1.0)
        self.assertEqual(m["iso_n2_match_frac"], 1.0)
        self.assertEqual(m["iso_n3_match_frac"], 1.0)

    def test_batch_that_changes_a_reply_is_reported(self):
        rec = si.run(self._serve(True), "m", 4, [2, 3], 8, 16)
        m = si.flat_metrics(rec)
        self.assertEqual(m["iso_solo_repeat_frac"], 1.0)
        self.assertLess(m["iso_n3_match_frac"], 1.0)
        self.assertLessEqual(m["iso_n3_prefix_min"], 10.0)

    def test_group_size_must_fit_prompts(self):
        with self.assertRaises(Exception):
            si.run("http://127.0.0.1:1", "m", 2, [3], 8, 16)


if __name__ == "__main__":
    unittest.main()
