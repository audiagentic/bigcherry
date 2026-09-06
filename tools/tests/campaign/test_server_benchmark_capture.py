"""Server cells retain failures and require observed clean teardown."""
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import call
from unittest.mock import patch

from bigcherry.campaign import benchmark
from bigcherry.campaign.benchmark import run_server_arm_capture
from bigcherry.tuning.server_runner import ShutdownResult
from bigcherry.source.identity import SourceAttestation


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
                )) + f"\nCMAKE_HOME_DIRECTORY:INTERNAL={self.root / 'live-source'}\n", encoding="utf-8",
            )
            (build / f"bigcherry-build-metadata-{binary.name}.json").write_text(
                json.dumps({"binary_hash": "digest", "runtime_artifacts": {"llama-server": "digest"},
                            "source_slice_id": "slice"}),
                encoding="utf-8",
            )
            self.arms.append({"name": name, "binary": str(binary), "mode": mode})
        self.base_config = {
            "schema_version": 1, "evidence_role": "production",
            "model": str(self.model), "server_args": ["-sm", "tensor"],
            "required_metrics": ["tg128_tps"], "bench_configs": "mtp-dual",
            "runner_root": str(self.runner), "repetitions": 1,
            "environment": {"HIP_VISIBLE_DEVICES": "0,1", "ROCR_VISIBLE_DEVICES": "0,1"},
            "expected_execution": {
                "backend": "ROCm", "architectures": ["gfx1100", "gfx1100"],
                "locators": ["0000:01:00.0", "0000:02:00.0"],
            },
            "arms": self.arms,
        }
        (self.root / "live-source").mkdir()
        (self.root / "live-source.metadata.json").write_text(
            json.dumps({"upstream_revision": "rev", "source_tree_oid": "tree",
                        "git_object_format": "sha1", "source_slice_id": "slice"}),
            encoding="utf-8",
        )

    def write_config(self, **updates):
        config = dict(self.base_config)
        config.update(updates)
        self.config.write_text(json.dumps(config), encoding="utf-8")

    def patches_for_preflight(self, *, instrumented=False):
        stack = ExitStack()
        stack.enter_context(patch.multiple(
            "bigcherry.build.builds",
            inspect_dispatch_build=lambda _path: {"issues": [], "instrumented": instrumented},
            binary_hash=lambda _path: "digest",
            resolve_runtime_artifacts=lambda path: [path],
        ))
        stack.enter_context(patch(
            "bigcherry.campaign.workers._source_attestation",
            return_value=SourceAttestation("rev", "tree", "sha1", "slice"),
        ))
        stack.enter_context(patch("bigcherry.campaign.workers._verify_source"))
        return stack

    def test_cli_server_config_routes_to_server_capture(self):
        self.write_config()
        from bigcherry.cli.main import main as public_main
        with patch("bigcherry.campaign.benchmark.run_server_comparison_capture", return_value=0) as capture, \
             patch("bigcherry.campaign.benchmark._run_arm") as generic:
            result = public_main(["ab-benchmark",
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

    def test_missing_source_home_directory_fails_before_server_launch(self):
        self.write_config()
        for arm in self.arms:
            build = Path(arm["binary"]).parent.parent
            cache = build / "CMakeCache.txt"
            cache.write_text(
                "\n".join(
                    line for line in cache.read_text(encoding="utf-8").splitlines()
                    if not line.startswith("CMAKE_HOME_DIRECTORY:")
                ) + "\n",
                encoding="utf-8",
            )
        with self.patches_for_preflight(), patch(
            "bigcherry.campaign.benchmark.run_server_arm_capture",
        ) as capture:
            with self.assertRaisesRegex(ValueError, "CMAKE_HOME_DIRECTORY is required"):
                benchmark.run_server_comparison_capture(
                    self.config, self.output, rounds=2, seed=0, settle_seconds=0
                )
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

    def test_live_source_is_reattested_before_each_cell_and_failure_stops_next_cell(self):
        from bigcherry.source.identity import SourceAttestation

        attestation = SourceAttestation("rev", "tree", "sha1", "slice")
        for arm in self.arms:
            build = Path(arm["binary"]).parent.parent
            cache = build / "CMakeCache.txt"
            cache.write_text(cache.read_text(), encoding="utf-8")
            metadata = build / "bigcherry-build-metadata-llama-server.json"
            metadata.write_text(json.dumps({
                "binary_hash": "digest", "runtime_artifacts": {"llama-server": "digest"},
                "source_slice_id": "slice",
            }), encoding="utf-8")
        self.write_config()
        captured = []
        verification_calls = []

        def verify(_root, _expected):
            verification_calls.append(call(_root, _expected))
            if len(verification_calls) == 4:
                raise ValueError("live source changed")

        with self.patches_for_preflight(), patch(
            "bigcherry.campaign.benchmark.run_server_arm_capture",
            side_effect=lambda **kwargs: captured.append(kwargs) or {
                "pair": kwargs["pair"] + 1, "mode": kwargs["side"],
                "position": kwargs["position"], "returncode": 0,
                "metrics": {"tg128_tps": 30.0},
            },
        ), patch("bigcherry.campaign.workers._source_attestation", return_value=attestation), patch(
            "bigcherry.campaign.workers._verify_source", side_effect=verify,
        ):
            self.assertEqual(benchmark.run_server_comparison_capture(
                self.config, self.output, rounds=2, seed=0, settle_seconds=0,
            ), 1)
        self.assertEqual(len(captured), 1)
        self.assertEqual(len(verification_calls), 4)
        summary = json.loads((self.output / "run.json").read_text())
        self.assertEqual(summary["runs"][-1]["source_attestation_error"], "live source changed")


class ServerExecutionAttestationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.expected = {
            "backend": "ROCm", "architectures": ["gfx1100", "gfx1100"],
            "locators": ["0000:01:00.0", "0000:02:00.0"],
        }

    def run_cell(self, startup, execution_evidence="required"):
        class Server:
            def __init__(self, **kwargs):
                self.host, self.port = "127.0.0.1", 4567
                self.last_shutdown = None
                self.log_path = kwargs["log_path"]

            def __enter__(self):
                self.log_path.write_text(startup, encoding="utf-8")
                return self

            def __exit__(self, *args):
                self.last_shutdown = ShutdownResult("http", True, False, 0)

        with patch("bigcherry.tuning.server_runner.ServerRunner", Server), patch(
            "bigcherry.campaign.bench_runner.run_bench_runner_server_bench",
            return_value={"tg128_tps": 30.0},
        ) as bench:
            result = run_server_arm_capture(
                binary=Path("server"), model=Path("model"), extra_args=(),
                output=self.root, pair=0, side="native", position=0, env={},
                bench_configs="mtp-dual", runner_root=Path("runner"),
                required_metrics=("tg128_tps",), expected_execution=self.expected,
                execution_evidence=execution_evidence,
            )
        return result, bench

    def test_matching_attestation_allows_bench(self):
        startup = (
            "I llama_prepare_model_devices: using device ROCm0 (AMD Radeon RX 7900 XTX) (0000:01:00.0)\n"
            "I llama_prepare_model_devices: using device ROCm1 (AMD Radeon RX 7900 XTX) (0000:02:00.0)\n"
            "D load_tensors: layer 0 assigned to device ROCm0\n"
        )
        result, bench = self.run_cell(startup)
        bench.assert_called_once()
        self.assertEqual(result["returncode"], 0)
        self.assertEqual(result["metrics"], {"tg128_tps": 30.0})
        self.assertEqual(result["execution_attestation"]["backend"], "ROCm")

    def test_wrong_physical_device_blocks_bench(self):
        startup = "using device ROCm0 (AMD Radeon RX 7900 XTX) (0000:03:00.0)\n"
        result, bench = self.run_cell(startup)
        bench.assert_not_called()
        self.assertEqual(result["returncode"], 1)
        self.assertNotIn("metrics", result)
        self.assertTrue(result["shutdown"]["requested"])

    def test_cpu_fallback_blocks_bench(self):
        result, bench = self.run_cell("failed to initialize ROCm: no ROCm-capable device is detected\n")
        bench.assert_not_called()
        self.assertEqual(result["returncode"], 1)
        self.assertNotIn("metrics", result)
        self.assertTrue(result["shutdown"]["requested"])

    def test_missing_attestation_blocks_bench(self):
        result, bench = self.run_cell("server is ready\n")
        bench.assert_not_called()
        self.assertEqual(result["returncode"], 1)
        self.assertNotIn("metrics", result)
        self.assertTrue(result["shutdown"]["requested"])

    def test_explicit_observation_collects_missing_evidence_without_admission(self):
        result, bench = self.run_cell("server is ready\n", "observe")
        bench.assert_called_once()
        self.assertEqual(result["returncode"], 0)
        self.assertFalse(result["performance_admitted"])
        self.assertEqual(result["execution_evidence_status"], "missing")
        self.assertIsNone(result["execution_attestation"])
        self.assertIn("physical-device execution evidence missing", result["admission_blockers"])

    def test_observation_does_not_ignore_cpu_fallback_or_wrong_devices(self):
        for startup in (
            "failed to initialize ROCm: no ROCm-capable device is detected\n",
            "using device ROCm0 (AMD) (0000:03:00.0)\n",
        ):
            with self.subTest(startup=startup):
                # Each cell retains a fresh artifact directory.
                self.root = self.root / "next"
                result, bench = self.run_cell(startup, "observe")
                bench.assert_not_called()
                self.assertEqual(result["returncode"], 1)
                self.assertNotIn("metrics", result)
