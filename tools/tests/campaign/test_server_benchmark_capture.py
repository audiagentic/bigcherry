"""Server cells retain failures and require observed clean teardown."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bigcherry.campaign import benchmark
from bigcherry.campaign.benchmark import run_server_arm_capture
from bigcherry.tuning.server_runner import ShutdownResult


class ServerCaptureTests(unittest.TestCase):
    def test_clean_and_forced_shutdown(self):
        for forced in (False, True):
            with self.subTest(forced=forced), tempfile.TemporaryDirectory() as tmp:
                class Server:
                    def __init__(self, **kwargs):
                        self.host, self.port = "127.0.0.1", 4567
                        self.last_shutdown = None

                    def __enter__(self):
                        return self

                    def __exit__(self, *args):
                        self.last_shutdown = ShutdownResult("http", True, forced, -9 if forced else 0)

                with patch("bigcherry.tuning.server_runner.ServerRunner", Server), patch(
                    "bigcherry.campaign.bench_runner.run_bench_runner_server_bench",
                    return_value={"tg512_tps": 30.0},
                ) as bench:
                    result = run_server_arm_capture(
                        binary=Path("llama-server"), model=Path("model.gguf"), extra_args=(),
                        output=Path(tmp), pair=0, side="native", position=1, env={},
                        bench_configs="tg512", runner_root=Path("runner"),
                        required_metrics=("tg512_tps",),
                    )
                self.assertEqual(result["returncode"], int(forced))
                self.assertEqual("metrics" in result, not forced)
                self.assertFalse(result["performance_admitted"])
                self.assertEqual(result["position"], 1)
                self.assertEqual(bench.call_args.kwargs["server_url"], "http://127.0.0.1:4567")
                persisted = json.loads((Path(tmp) / "pair-001-native/cell.json").read_text())
                self.assertEqual(persisted["shutdown"]["forced"], forced)

    def test_rejects_missing_expected_metrics_before_launch(self):
        with patch("bigcherry.tuning.server_runner.ServerRunner") as server:
            with self.assertRaisesRegex(ValueError, "expected metrics"):
                run_server_arm_capture(
                    binary=Path("server"), model=Path("model"), extra_args=(),
                    output=Path("unused"), pair=0, side="native", position=0,
                    env={}, bench_configs="full", runner_root=Path("runner"), required_metrics=(),
                )
        server.assert_not_called()


class ServerComparisonCaptureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = self.root / "server.json"
        self.output = self.root / "capture"
        self.model = self.root / "model.gguf"
        self.model.write_bytes(b"model")
        self.runner = self.root / "runner"
        (self.runner / "bench/config").mkdir(parents=True)
        (self.runner / "bench/run_bench.py").write_text("# fixture", encoding="utf-8")
        self.arms = []
        for name, mode in (("stock", "stock"), ("native", "native")):
            build = self.root / name / "build"
            binary = build / "bin" / "llama-server"
            binary.parent.mkdir(parents=True)
            binary.write_bytes(name.encode())
            (build / "CMakeCache.txt").write_text(
                "\n".join(f"{key}:STRING=value" for key in (
                    "CMAKE_BUILD_TYPE", "CMAKE_C_COMPILER", "CMAKE_CXX_COMPILER",
                    "GGML_HIP", "GGML_HIP_RCCL", "AMDGPU_TARGETS",
                )), encoding="utf-8",
            )
            (build / f"bigcherry-build-metadata-{binary.name}.json").write_text(
                json.dumps({"binary_hash": "digest", "runtime_artifacts": {"llama-server": "digest"}}),
                encoding="utf-8",
            )
            self.arms.append({"name": name, "binary": str(binary), "mode": mode})
        self.base_config = {
            "schema_version": 1, "evidence_role": "production",
            "model": str(self.model), "server_args": ["-sm", "tensor"],
            "required_metrics": ["tg128_tps"], "bench_configs": "mtp-dual",
            "runner_root": str(self.runner), "repetitions": 1,
            "environment": {"HIP_VISIBLE_DEVICES": "0,1", "ROCR_VISIBLE_DEVICES": "0,1"},
            "arms": self.arms,
        }

    def write_config(self, **updates):
        config = dict(self.base_config)
        config.update(updates)
        self.config.write_text(json.dumps(config), encoding="utf-8")

    def patches_for_preflight(self, *, instrumented=False):
        return patch.multiple(
            "bigcherry.build.builds",
            inspect_dispatch_build=lambda _path: {"issues": [], "instrumented": instrumented},
            binary_hash=lambda _path: "digest",
            resolve_runtime_artifacts=lambda path: [path],
        )

    def test_cli_server_config_routes_to_server_capture(self):
        self.write_config()
        with patch("bigcherry.campaign.benchmark.run_server_comparison_capture", return_value=0) as capture, \
             patch("bigcherry.campaign.benchmark._run_arm") as generic:
            result = benchmark.main([
                "--server-config", str(self.config), "--output", str(self.output),
                "--pairs", "2", "--settle-seconds", "0",
            ])
        self.assertEqual(result, 0)
        capture.assert_called_once_with(self.config, self.output, rounds=2, seed=0, settle_seconds=0.0)
        generic.assert_not_called()

    def test_balanced_two_arm_schedule_executes_both_orders_and_positions(self):
        self.write_config()
        captured = []
        def fake_capture(**kwargs):
            captured.append(kwargs)
            return {"pair": kwargs["pair"] + 1, "mode": kwargs["side"], "position": kwargs["position"], "returncode": 0, "metrics": {"tg128_tps": 30.0}}
        with self.patches_for_preflight(), patch(
            "bigcherry.campaign.benchmark.run_server_arm_capture", side_effect=fake_capture,
        ):
            self.assertEqual(benchmark.run_server_comparison_capture(self.config, self.output, rounds=2, seed=0, settle_seconds=0), 0)
        self.assertEqual([(item["pair"] + 1, item["side"], item["position"]) for item in captured], [
            (1, "stock", 0), (1, "native", 1), (2, "native", 0), (2, "stock", 1),
        ])
        summary = json.loads((self.output / "run.json").read_text())
        self.assertEqual(summary["performance_admitted"], False)
        self.assertEqual(len(summary["runs"]), 4)
        self.assertIn("exploratory_comparisons", summary)

    def test_production_role_rejects_instrumented_build(self):
        self.write_config()
        with self.patches_for_preflight(instrumented=True), patch(
            "bigcherry.campaign.benchmark.run_server_arm_capture",
        ) as capture:
            with self.assertRaisesRegex(ValueError, "instrumentation disagrees"):
                benchmark.run_server_comparison_capture(self.config, self.output, rounds=2, seed=0, settle_seconds=0)
        capture.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_failed_cell_persists_without_exploratory_comparison(self):
        self.write_config()
        with self.patches_for_preflight(), patch(
            "bigcherry.campaign.benchmark.run_server_arm_capture",
            return_value={"pair": 1, "mode": "stock", "position": 0, "returncode": 1},
        ):
            self.assertEqual(benchmark.run_server_comparison_capture(self.config, self.output, rounds=2, seed=0, settle_seconds=0), 1)
        summary = json.loads((self.output / "run.json").read_text())
        self.assertEqual(len(summary["runs"]), 1)
        self.assertNotIn("exploratory_comparisons", summary)
