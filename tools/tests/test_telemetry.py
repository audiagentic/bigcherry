from __future__ import annotations

import io
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout

from bigcherry.telemetry import console_telemetry, summarize_launch


class TelemetryTests(unittest.TestCase):
    def test_summarize_launch_never_includes_raw_argv(self) -> None:
        summary = summarize_launch(["echo", "SECRET_TOKEN=abc123"], {"A": "1", "B": "2"})
        self.assertEqual(summary.command_name, "echo")
        self.assertEqual(summary.argv_count, 2)
        self.assertEqual(summary.env_key_count, 2)
        # digest is derived, not the raw text
        self.assertNotIn("SECRET_TOKEN", summary.argv_digest)

    def test_telemetry_goes_to_stderr_not_stdout(self) -> None:
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            with console_telemetry(session_id="s1", command=["cmd", "arg1", "arg2"]) as state:
                print("machine readable output")
                state["returncode"] = 0
                state["output_summary"] = "ok"
        self.assertEqual(out.getvalue().strip(), "machine readable output")
        self.assertIn("launch", err.getvalue())
        self.assertIn("completion", err.getvalue())
        self.assertIn("returncode=0", err.getvalue())

    def test_default_mode_does_not_leak_raw_argv_values(self) -> None:
        err = io.StringIO()
        with redirect_stderr(err):
            with console_telemetry(session_id="s2", command=["cmd", "--password", "hunter2"]) as state:
                state["returncode"] = 0
        self.assertNotIn("hunter2", err.getvalue())

    def test_show_argv_true_reveals_argv_for_trusted_callers(self) -> None:
        err = io.StringIO()
        with redirect_stderr(err):
            with console_telemetry(session_id="s3", command=["cmake", "--build", "."], show_argv=True) as state:
                state["returncode"] = 0
        self.assertIn("cmake --build .", err.getvalue())

    def test_completion_line_emitted_even_on_exception(self) -> None:
        err = io.StringIO()
        with self.assertRaises(RuntimeError):
            with redirect_stderr(err):
                with console_telemetry(session_id="s4", command=["cmd"]) as state:
                    raise RuntimeError("boom")
        self.assertIn("completion", err.getvalue())


if __name__ == "__main__":
    unittest.main()
