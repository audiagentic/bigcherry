"""Offline tests for retained server-bench invocation evidence."""
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bigcherry.campaign.bench_runner import BenchRunnerError, run_bench_runner_server_bench


class BenchRunnerEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "bench").mkdir()
        (self.root / "bench/run_bench.py").write_text("# fixture", encoding="utf-8")
        self.output = self.root / "cell"

    def run_cell(self):
        return run_bench_runner_server_bench(
            server_url="http://127.0.0.1:4567", bench_configs="tg512",
            model_label="another-model", runner_root=self.root,
            evidence_dir=self.output, required_metrics=("tg512_tps",),
        )

    def test_retains_request_and_complete_streams(self):
        result = subprocess.CompletedProcess([], 0, "Extracted Results:\n tg512_tps: 30.5\n", "notice")
        with patch("bigcherry.campaign.bench_runner.subprocess.run", return_value=result):
            self.assertEqual(self.run_cell(), {"tg512_tps": 30.5})
        request = json.loads((self.output / "request.json").read_text())
        self.assertIn("another-model", request["command"])
        self.assertIn("server-bench", request["command"])
        self.assertEqual((self.output / "stderr.log").read_text(), "notice")
        self.assertEqual(len(request["runner_sha256"]), 64)

    def test_missing_metric_fails_after_retaining_output(self):
        result = subprocess.CompletedProcess([], 0, "Extracted Results:\n pp512_tps: 300\n", "")
        with patch("bigcherry.campaign.bench_runner.subprocess.run", return_value=result):
            with self.assertRaisesRegex(BenchRunnerError, "omitted required"):
                self.run_cell()
        self.assertTrue((self.output / "stdout.log").is_file())

    def test_timeout_retains_partial_bytes(self):
        timeout = subprocess.TimeoutExpired([], 300, output=b"partial", stderr=b"error")
        with patch("bigcherry.campaign.bench_runner.subprocess.run", side_effect=timeout):
            with self.assertRaisesRegex(BenchRunnerError, "timed out"):
                self.run_cell()
        self.assertEqual((self.output / "stdout.log").read_text(), "partial")
        self.assertTrue(json.loads((self.output / "exit.json").read_text())["timed_out"])

    def test_existing_evidence_is_never_overwritten(self):
        self.output.mkdir()
        with patch("bigcherry.campaign.bench_runner.subprocess.run") as run:
            with self.assertRaises(FileExistsError):
                self.run_cell()
        run.assert_not_called()

    def test_zero_throughput_is_not_valid_measurement(self):
        result = subprocess.CompletedProcess([], 0, "Extracted Results:\n tg512_tps: 0\n", "")
        with patch("bigcherry.campaign.bench_runner.subprocess.run", return_value=result):
            with self.assertRaisesRegex(BenchRunnerError, "non-positive"):
                self.run_cell()
