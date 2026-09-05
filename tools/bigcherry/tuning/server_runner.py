"""HI130: real llama-server process lifecycle, extracted from
e2e_smoke_campaign.py's Campaign class so tune-campaign's workflow.py (and
any other future orchestrator) can drive a real server without duplicating
this logic or depending on e2e_smoke_campaign's own pre-built-binary,
non-campaign-engine assumptions.

Real subprocess + real HTTP against a real llama-server -- there is no
mock/simulation mode. Callers that need to avoid touching a GPU should not
call this module at all.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


def _free_port(host: str) -> int:
    """Bind port 0 and let the OS choose, then release it.

    There is an unavoidable race between releasing and llama-server binding,
    but it is far smaller than the certainty of colliding with a long-lived
    service on a fixed port -- which is what actually happened here.
    """
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


class ServerError(RuntimeError):
    pass


class ServerRunner:
    """One real llama-server process, launched, health-checked, driven
    with real HTTP requests, and shut down cleanly.

    ``LLAMA_SERVER_ENABLE_SHUTDOWN=1`` + the opt-in ``/shutdown`` route is
    used for teardown rather than a bare process kill: a plain kill skips
    backend teardown and silently discards buffered HIP autotune
    measurements (found the hard way in an earlier session -- an early
    version of the e2e smoke campaign used a bare kill and every tune run
    came back empty).
    """

    def __init__(
        self, *, binary: Path, model: Path, host: str = "127.0.0.1",
        port: int | None = None,
        extra_args: tuple[str, ...] = (), env_overrides: dict[str, str] | None = None,
        env_unset: tuple[str, ...] = (),
        log_path: Path | None = None, command_prefix: tuple[str, ...] = (),
    ):
        self.binary = binary
        self.model = model
        self.host = host
        # port=None picks a FREE port instead of defaulting to 8080.
        #
        # 8080 was hardcoded, and on the tuning host llama-swap -- the
        # production inference service -- already owns it. A real
        # `tune-campaign` run died at its first stage with "exiting due to
        # HTTP server error", which is what a bind collision looks like from
        # llama-server. The tuner must be able to run alongside production
        # without either interfering with the other, and without an operator
        # remembering to pass a port.
        self.port = port if port is not None else _free_port(host)
        self.extra_args = extra_args
        self.env_overrides = dict(env_overrides or {})
        # HI143 (gpt review, 2026-08-29): the launched process otherwise
        # inherits the FULL ambient environment -- a caller's leftover
        # GGML_HIP_FORCE_CANDIDATE(_STRICT)/DISPATCH_DB/CACHE/COVERAGE from
        # an unrelated earlier command could silently contaminate a run
        # that never meant to set them. Names listed here are removed from
        # the inherited environment before env_overrides is applied.
        self.env_unset = tuple(env_unset)
        self.log_path = log_path
        # PROF01: lets a profiler (e.g. rocprofv3) wrap the real server
        # launch without duplicating ServerRunner's own lifecycle/health/
        # shutdown handling. The prefix is inserted before the binary path,
        # e.g. ("rocprofv3", "--sys-trace", "--rccl-trace", "-d", str(outdir),
        # "--") -- the caller supplies its own "--" separator since the
        # exact flag comes before the target command for every profiler
        # checked (rocprofv3, perf record).
        self.command_prefix = command_prefix
        self._proc: subprocess.Popen | None = None

    def _base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def launch(self) -> None:
        if self._proc is not None:
            raise ServerError("server already launched")
        env = dict(os.environ)
        for name in self.env_unset:
            env.pop(name, None)
        env.update(self.env_overrides)
        env["LLAMA_SERVER_ENABLE_SHUTDOWN"] = "1"
        args = [
            *self.command_prefix,
            str(self.binary), "-m", str(self.model),
            "--port", str(self.port), "--host", self.host,
            *self.extra_args,
        ]
        if self.log_path is not None:
            with self.log_path.open("w", encoding="utf-8") as log_file:
                self._proc = subprocess.Popen(args, env=env, stdout=log_file, stderr=subprocess.STDOUT)
        else:
            self._proc = subprocess.Popen(args, env=env)

    def wait_healthy(self, timeout_s: int = 180) -> None:
        if self._proc is None:
            raise ServerError("server not launched")
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self._proc.poll() is not None:
                tail = self._log_tail()
                raise ServerError(
                    f"server process exited (code {self._proc.returncode}) "
                    f"before becoming healthy: {tail}"
                )
            try:
                with urllib.request.urlopen(f"{self._base_url()}/health", timeout=2) as resp:
                    if resp.status == 200:
                        return
            except (urllib.error.URLError, OSError, TimeoutError):
                pass
            time.sleep(1)
        raise ServerError(f"server did not become healthy within {timeout_s}s")

    def post_json(self, path: str, payload: dict, timeout_s: int = 300) -> dict:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self._base_url()}{path}", data=body,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                raw = resp.read()
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            raise ServerError(f"POST {path} failed: {exc}") from exc
        if not raw:
            # /shutdown and similar routes legitimately return an empty
            # body -- not every real endpoint replies with JSON.
            return {}
        return json.loads(raw.decode("utf-8"))

    def run_completion(self, prompt: str, *, n_predict: int = 96, timeout_s: int = 300) -> dict:
        return self.post_json(
            "/completion", {"prompt": prompt, "n_predict": n_predict}, timeout_s=timeout_s,
        )

    def shutdown(self, timeout_s: int = 90) -> None:
        if self._proc is None:
            return
        try:
            self.post_json("/shutdown", {})
        except ServerError:
            pass
        try:
            self._proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait(timeout=30)
        self._proc = None

    def _log_tail(self, n: int = 15) -> str:
        if self.log_path is None or not self.log_path.is_file():
            return "(no log)"
        lines = self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-n:])

    def __enter__(self) -> "ServerRunner":
        self.launch()
        try:
            self.wait_healthy()
        except BaseException:
            # HI143 (gpt review, 2026-08-29): without this, a health-check
            # timeout/failure left the just-launched process (and its real
            # GPU allocation) running forever -- __exit__ is never called
            # when __enter__ itself raises, so this was the only place that
            # could clean it up.
            self.shutdown()
            raise
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown()
