"""A tiny stand-in for an engine server, used only by test_cpu_smoke.py.

It answers /health, /completion (llama.cpp shape), /v1/models and /v1/completions (OpenAI shape) and stops on
POST /shutdown. ``--nondeterministic`` makes every answer differ; ``--gpu-log`` prints a device-init line;
``--empty`` answers with empty text. It prints the GPU-selection variables and any GGML_ variables it inherited so a
test can see the environment the runner built.
"""

import argparse
import hashlib
import itertools
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

parser = argparse.ArgumentParser()
parser.add_argument("--model")
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int)
parser.add_argument("--nondeterministic", action="store_true")
parser.add_argument("--gpu-log", action="store_true")
parser.add_argument("--empty", action="store_true")
args, _unknown = parser.parse_known_args()

counter = itertools.count()
for name in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
    print(f"env {name}={os.environ.get(name)!r}", flush=True)
print("env GGML_FOO=" + repr(os.environ.get("GGML_FOO")), flush=True)
if args.gpu_log:
    print("ggml_cuda_init: found 1 ROCm devices:", flush=True)


def text_for(prompt: str) -> str:
    if args.empty:
        return ""
    base = "tok-" + hashlib.sha1(prompt.encode()).hexdigest()[:8]
    return f"{base}-{next(counter)}" if args.nondeterministic else base


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def _send(self, document):
        body = json.dumps(document).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send({"status": "ok"})
        elif self.path == "/v1/models":
            self._send({"data": [{"id": "tiny"}]})
        else:
            self.send_error(404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        if self.path == "/completion":
            self._send({"content": text_for(body["prompt"])})
        elif self.path == "/v1/completions":
            assert body.get("model") == "tiny", body
            self._send({"choices": [{"text": text_for(body["prompt"])}]})
        elif self.path == "/shutdown":
            self._send({})
            threading.Thread(target=server.shutdown, daemon=True).start()
        else:
            self.send_error(404)


server = HTTPServer((args.host, args.port), Handler)
server.serve_forever()
