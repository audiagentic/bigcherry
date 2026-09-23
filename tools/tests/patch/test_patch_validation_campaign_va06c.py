"""VA06: the shared workload-metric registration and bench-runner parsing.

The RD73 legacy CLI path (run_rd73_* lanes, evaluate_rd73_* evaluators and
run_rd73_contract_qualification in validation_campaign.py) was retired in
PA43 after patch 1233_rd73_stable_graph_cache_key was rejected; RD73's lanes
live only in its patch-local producer. What remains here is generic.
Hardware-free: subprocess.run is faked.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.experiment import execution as ee  # noqa: E402
from bigcherry.campaign import bench_runner  # noqa: E402


class WorkloadMetricRegistrationTests(unittest.TestCase):
    def test_mtp_verify_maps_to_mtp_wall_tps(self) -> None:
        self.assertEqual(ee.metric_for_workload("mtp_verify"), "mtp_wall_tps")


class RunBenchRunnerServerBenchTests(unittest.TestCase):
    def setUp(self) -> None:
        self._real_run = subprocess.run
        self._tmp = tempfile.TemporaryDirectory()
        self.runner_root = Path(self._tmp.name)
        (self.runner_root / "bench").mkdir()
        (self.runner_root / "bench" / "run_bench.py").write_text("", encoding="utf-8")

    def tearDown(self) -> None:
        subprocess.run = self._real_run
        self._tmp.cleanup()

    def test_parses_aggregated_results_block(self) -> None:
        def fake_run(command, **kwargs):
            class _Result:
                returncode = 0
                stdout = (
                    "some header text\n"
                    "\nAggregated Results (1 test(s)):\n"
                    "            tg128_tps: 42.5\n"
                )
                stderr = ""

            return _Result()

        subprocess.run = fake_run
        metrics = bench_runner.run_bench_runner_server_bench(
            server_url="http://127.0.0.1:18080",
            bench_configs="tg128",
            runner_root=self.runner_root,
        )
        self.assertEqual(metrics["tg128_tps"], 42.5)

    def test_parses_extracted_results_block(self) -> None:
        # Real hardware finding: server-bench mode (used by RD73's decode
        # control lane) prints "Extracted Results", not "Aggregated
        # Results" (bench/runners/server_base.py, confirmed against a
        # real Brutus run) -- an earlier draft only handled the latter.
        def fake_run(command, **kwargs):
            class _Result:
                returncode = 0
                stdout = (
                    "some header text\n"
                    "\nExtracted Results (1 config(s)):\n"
                    "             tg128_tps: 27.29\n"
                    "          tg128_stddev: 0.00\n"
                )
                stderr = ""

            return _Result()

        subprocess.run = fake_run
        metrics = bench_runner.run_bench_runner_server_bench(
            server_url="http://127.0.0.1:18082",
            bench_configs="tg128",
            runner_root=self.runner_root,
        )
        self.assertEqual(metrics["tg128_tps"], 27.29)
        self.assertNotIn("tg128_stddev", metrics)

    def test_missing_runner_script_fails_closed(self) -> None:
        with self.assertRaises(bench_runner.BenchRunnerError):
            bench_runner.run_bench_runner_server_bench(
                server_url="http://127.0.0.1:18080",
                bench_configs="tg128",
                runner_root=Path("/nonexistent"),
            )

    def test_nonzero_exit_fails_closed(self) -> None:
        def fake_run(command, **kwargs):
            class _Result:
                returncode = 1
                stdout = "error"
                stderr = "boom"

            return _Result()

        subprocess.run = fake_run
        with self.assertRaises(bench_runner.BenchRunnerError):
            bench_runner.run_bench_runner_server_bench(
                server_url="http://127.0.0.1:18080",
                bench_configs="tg128",
                runner_root=self.runner_root,
            )

    def test_no_parseable_metrics_fails_closed(self) -> None:
        def fake_run(command, **kwargs):
            class _Result:
                returncode = 0
                stdout = "nothing useful here"
                stderr = ""

            return _Result()

        subprocess.run = fake_run
        with self.assertRaises(bench_runner.BenchRunnerError):
            bench_runner.run_bench_runner_server_bench(
                server_url="http://127.0.0.1:18080",
                bench_configs="tg128",
                runner_root=self.runner_root,
            )


if __name__ == "__main__":
    unittest.main()
