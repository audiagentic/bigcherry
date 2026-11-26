from __future__ import annotations

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from bigcherry.bench.served_model import served_model


class _Models(BaseHTTPRequestHandler):
    listing: dict = {}

    def do_GET(self):
        body = json.dumps(type(self).listing).encode()
        self.send_response(200 if self.path == "/v1/models" else 404)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class ServedModel(unittest.TestCase):
    def setUp(self):
        served_model.cache_clear()
        self.server = HTTPServer(("127.0.0.1", 0), _Models)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def test_first_listed_model_is_the_served_one(self):
        _Models.listing = {"data": [{"id": "tiny"}, {"id": "other"}]}
        self.assertEqual(served_model(self.url), "tiny")

    def test_a_server_that_lists_no_model_is_an_error(self):
        _Models.listing = {"data": []}
        with self.assertRaises(RuntimeError):
            served_model(self.url)


if __name__ == "__main__":
    unittest.main()
