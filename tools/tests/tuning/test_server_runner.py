"""HI130: ServerRunner lifecycle -- launch/health-check/HTTP/shutdown."""

from __future__ import annotations

import http.server
import json
import signal
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.tuning.server_runner import ServerError, ServerRunner  # noqa: E402


class _FakeServerHandler(http.server.BaseHTTPRequestHandler):
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
        body = self.rfile.read(length) if length else b"{}"
        if self.path == "/completion":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"content": "ok", "echo": json.loads(body)}).encode())
        elif self.path == "/shutdown":
            self.send_response(200)
            self.end_headers()
            threading.Thread(target=self.server.shutdown, daemon=True).start()
        else:
            self.send_response(404)
            self.end_headers()


class ServerRunnerHttpTests(unittest.TestCase):
    """Real HTTP against a real local server -- only subprocess.Popen (the
    actual llama-server process) is mocked, since spawning a real GPU
    binary is out of scope for an offline unit test."""

    def setUp(self):
        self.httpd = http.server.HTTPServer(("127.0.0.1", 0), _FakeServerHandler)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self._stop_httpd)

    def _stop_httpd(self):
        try:
            self.httpd.shutdown()
        except Exception:
            pass

    def _runner(self) -> ServerRunner:
        return ServerRunner(
            binary=Path("fake-binary"), model=Path("fake-model.gguf"),
            host="127.0.0.1", port=self.port,
        )

    def test_wait_healthy_succeeds_against_a_real_health_endpoint(self):
        runner = self._runner()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        with patch("bigcherry.tuning.server_runner.subprocess.Popen", return_value=mock_proc):
            runner.launch()
            runner.wait_healthy(timeout_s=5)

    def test_run_completion_returns_real_parsed_json(self):
        runner = self._runner()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        with patch("bigcherry.tuning.server_runner.subprocess.Popen", return_value=mock_proc):
            runner.launch()
            runner.wait_healthy(timeout_s=5)
            result = runner.run_completion("hello", n_predict=8)
        self.assertEqual(result["content"], "ok")
        self.assertEqual(result["echo"]["prompt"], "hello")
        self.assertEqual(result["echo"]["n_predict"], 8)

    def test_shutdown_posts_shutdown_and_waits_for_process(self):
        runner = self._runner()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        with patch("bigcherry.tuning.server_runner.subprocess.Popen", return_value=mock_proc):
            runner.launch()
            runner.wait_healthy(timeout_s=5)
            runner.shutdown()
        mock_proc.wait.assert_called()

    def test_context_manager_launches_and_shuts_down(self):
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        with patch("bigcherry.tuning.server_runner.subprocess.Popen", return_value=mock_proc):
            with self._runner() as runner:
                result = runner.run_completion("hi")
                self.assertEqual(result["content"], "ok")
        mock_proc.wait.assert_called()


class ServerRunnerFailureTests(unittest.TestCase):
    def test_shutdown_preserves_forced_exit_instead_of_implying_success(self):
        runner = ServerRunner(binary=Path("x"), model=Path("m"), port=1)
        proc = MagicMock()
        proc.wait.side_effect = [subprocess.TimeoutExpired("server", 1), -9]
        runner._proc = proc
        with patch.object(runner, "post_json", return_value={}):
            result = runner.shutdown(timeout_s=1)
        self.assertTrue(result.requested)
        self.assertTrue(result.forced)
        self.assertFalse(result.clean)
        self.assertEqual(result.returncode, -9)
        self.assertIs(runner.shutdown(), result)
        proc.kill.assert_called_once()

    def test_http_failure_and_nonzero_exit_are_not_clean(self):
        for returncode in (0, 1):
            with self.subTest(returncode=returncode):
                runner = ServerRunner(binary=Path("x"), model=Path("m"), port=1)
                runner._proc = MagicMock()
                runner._proc.wait.return_value = returncode
                with patch.object(runner, "post_json", side_effect=ServerError("404")):
                    result = runner.shutdown(timeout_s=1)
                self.assertFalse(result.clean)
                self.assertFalse(result.requested)
                self.assertEqual(result.error, "404")

    def test_successful_http_exit_is_recorded(self):
        runner = ServerRunner(binary=Path("x"), model=Path("m"), port=1)
        runner._proc = MagicMock()
        runner._proc.wait.return_value = 0
        with patch.object(runner, "post_json", return_value={}):
            result = runner.shutdown(timeout_s=1)
        self.assertTrue(result.clean)
        self.assertFalse(result.forced)

    def test_stock_sigint_uses_upstream_signal_handler(self):
        with patch("bigcherry.tuning.server_runner.os.name", "posix"):
            runner = ServerRunner(binary=Path("x"), model=Path("m"), port=1,
                                  shutdown_method="sigint")
        proc = MagicMock()
        proc.wait.return_value = 0
        runner._proc = proc
        with patch.object(runner, "post_json") as post:
            result = runner.shutdown(timeout_s=1)
        post.assert_not_called()
        proc.send_signal.assert_called_once_with(signal.SIGINT)
        self.assertTrue(result.clean)

    def test_signal_shutdown_rejects_wrappers_and_windows(self):
        binary, model = Path("x"), Path("m")
        with patch("bigcherry.tuning.server_runner.os.name", "nt"):
            with self.assertRaisesRegex(ValueError, "unwrapped POSIX"):
                ServerRunner(binary=binary, model=model, shutdown_method="sigint")
        with patch("bigcherry.tuning.server_runner.os.name", "posix"):
            with self.assertRaisesRegex(ValueError, "unwrapped POSIX"):
                ServerRunner(binary=binary, model=model, shutdown_method="sigint",
                             command_prefix=("profiler",))

    def test_wait_healthy_raises_when_process_exits_early(self):
        runner = ServerRunner(binary=Path("x"), model=Path("m.gguf"), port=1)
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 1
        mock_proc.returncode = 1
        with patch("bigcherry.tuning.server_runner.subprocess.Popen", return_value=mock_proc):
            runner.launch()
            with self.assertRaises(ServerError):
                runner.wait_healthy(timeout_s=2)

    def test_wait_healthy_raises_on_timeout_when_nothing_ever_answers(self):
        # Port 1 is a privileged, essentially-never-listening port -- health
        # checks will keep failing to connect until timeout.
        runner = ServerRunner(binary=Path("x"), model=Path("m.gguf"), port=1)
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        with patch("bigcherry.tuning.server_runner.subprocess.Popen", return_value=mock_proc):
            runner.launch()
            with self.assertRaises(ServerError):
                runner.wait_healthy(timeout_s=2)

    def test_context_manager_shuts_down_process_when_wait_healthy_raises(self):
        # HI143 (gpt review, 2026-08-29): __enter__ used to call launch()
        # then wait_healthy() with no cleanup -- a health-check failure
        # left the just-launched process (and its real GPU allocation)
        # running forever, since __exit__ is never invoked when __enter__
        # itself raises. wait_healthy() is patched to fail immediately
        # (rather than relying on its real default 180s timeout) so this
        # test stays fast.
        runner = ServerRunner(binary=Path("x"), model=Path("m.gguf"), port=1)
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        with (
            patch("bigcherry.tuning.server_runner.subprocess.Popen", return_value=mock_proc),
            patch.object(runner, "wait_healthy", side_effect=ServerError("never became healthy")),
        ):
            with self.assertRaises(ServerError):
                with runner:
                    pass  # never reached -- wait_healthy() raises first
        mock_proc.wait.assert_called()

    def test_launch_twice_raises(self):
        runner = ServerRunner(binary=Path("x"), model=Path("m.gguf"), port=2)
        mock_proc = MagicMock()
        with patch("bigcherry.tuning.server_runner.subprocess.Popen", return_value=mock_proc):
            runner.launch()
            with self.assertRaises(ServerError):
                runner.launch()

    def test_log_tail_used_in_error_when_log_path_given(self):
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "server.log"
            runner = ServerRunner(
                binary=Path("x"), model=Path("m.gguf"), port=1, log_path=log_path,
            )
            mock_proc = MagicMock()
            mock_proc.poll.return_value = 1
            mock_proc.returncode = 1
            with patch("bigcherry.tuning.server_runner.subprocess.Popen", return_value=mock_proc):
                runner.launch()
                # launch() truncates the log (a real subprocess would write
                # here); simulate the process having written its crash
                # reason before exiting.
                log_path.write_text("line1\nline2\ncrash reason here\n", encoding="utf-8")
                with self.assertRaisesRegex(ServerError, "crash reason here"):
                    runner.wait_healthy(timeout_s=2)


if __name__ == "__main__":
    unittest.main()
