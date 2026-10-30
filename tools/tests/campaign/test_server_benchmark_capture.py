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


class ServerAttestationBindingTests(unittest.TestCase):
    def test_binding_rejects_timed_argument_or_environment_drift(self):
        binding = {
            "binary": "server", "binary_sha256": None,
            "model": "model", "model_sha256": None,
            "common_server_args": ["-sm", "none"],
            "environment": {"HIP_VISIBLE_DEVICES": "0"},
        }
        with self.assertRaisesRegex(ValueError, "common_server_args"):
            benchmark._verify_attestation_binding(
                binding, binary=Path("server"), model=Path("model"),
                extra_args=("-sm", "tensor"), env={"HIP_VISIBLE_DEVICES": "0"},
            )
        with self.assertRaisesRegex(ValueError, "environment"):
            benchmark._verify_attestation_binding(
                binding, binary=Path("server"), model=Path("model"),
                extra_args=("-sm", "none"), env={"HIP_VISIBLE_DEVICES": "1"},
            )


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

                    def draft_stats(self):
                        return None

                    def __exit__(self, *args):
                        self.last_shutdown = ShutdownResult("http", True, forced, -9 if forced else 0)

                with patch("bigcherry.tuning.server_runner.ServerRunner", Server), patch(
                    "bigcherry.campaign.bench_runner.run_bench_runner_server_bench",
                    return_value={"tg512_tps": 30.0},
                ) as bench:
                    result = run_server_arm_capture(engine="llamacpp", settings={}, health_timeout_s=180,
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
                run_server_arm_capture(engine="llamacpp", settings={}, health_timeout_s=180,
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
            "schema_version": 1, "engine": "llamacpp", "evidence_role": "production",
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
        advisories = json.loads((self.output / "advisories.json").read_text())
        self.assertIn("AB_NOT_ADMITTED", [item["id"] for item in advisories["findings"]])

    def test_arm_environment_admits_allreduce_controls_but_not_topology(self):
        captured = []
        def fake_capture(**kwargs):
            captured.append(kwargs)
            return {"pair": kwargs["pair"] + 1, "mode": kwargs["side"], "position": kwargs["position"], "returncode": 0, "metrics": {"tg128_tps": 30.0}}
        arms = [dict(self.arms[0], environment={"GGML_CUDA_ALLREDUCE": "internal", "GGML_CUDA_AR_COPY_THRESHOLD": "65536",
                                                "NCCL_PROTO": "LL"}), self.arms[1]]
        self.write_config(arms=arms)
        with self.patches_for_preflight(), patch(
            "bigcherry.campaign.benchmark.run_server_arm_capture", side_effect=fake_capture,
        ):
            self.assertEqual(benchmark.run_server_comparison_capture(self.config, self.output, rounds=2, seed=0, settle_seconds=0), 0)
        env = next(item["env"] for item in captured if item["side"] == self.arms[0]["name"])
        self.assertEqual(env["GGML_CUDA_ALLREDUCE"], "internal")
        self.assertEqual(env["GGML_CUDA_AR_COPY_THRESHOLD"], "65536")
        self.assertEqual(env["NCCL_PROTO"], "LL")
        self.write_config(arms=[dict(self.arms[0], environment={"BIGCHERRY_PATCH_TRACE": "1"}), self.arms[1]])
        with self.patches_for_preflight(), self.assertRaisesRegex(ValueError, "not tracing"):
            benchmark.run_server_comparison_capture(self.config, self.output / "trace", rounds=2, seed=0, settle_seconds=0)
        self.write_config(arms=[dict(self.arms[0], environment={"HIP_VISIBLE_DEVICES": "0"}), self.arms[1]])
        with self.patches_for_preflight(), self.assertRaisesRegex(ValueError, "topology belongs"):
            benchmark.run_server_comparison_capture(self.config, self.output / "bad", rounds=2, seed=0, settle_seconds=0)

    def test_arm_server_args_are_appended_to_that_arm_only(self):
        captured = []
        def fake_capture(**kwargs):
            captured.append(kwargs)
            return {"pair": kwargs["pair"] + 1, "mode": kwargs["side"], "position": kwargs["position"], "returncode": 0, "metrics": {"tg128_tps": 30.0}}
        arms = [dict(self.arms[0], server_args=["--spec-draft-n-min-adaptive", "1"]), self.arms[1]]
        self.write_config(arms=arms)
        with self.patches_for_preflight(), patch(
            "bigcherry.campaign.benchmark.run_server_arm_capture", side_effect=fake_capture,
        ):
            self.assertEqual(benchmark.run_server_comparison_capture(self.config, self.output, rounds=2, seed=0, settle_seconds=0), 0)
        by_side = {item["side"]: item["extra_args"] for item in captured}
        self.assertEqual(by_side[self.arms[0]["name"]], ("-sm", "tensor", "--spec-draft-n-min-adaptive", "1"))
        self.assertEqual(by_side[self.arms[1]["name"]], ("-sm", "tensor"))
        self.write_config(arms=[dict(self.arms[0], server_args=["--port", "1"]), self.arms[1]])
        with self.patches_for_preflight(), self.assertRaisesRegex(ValueError, "managed model or endpoint"):
            benchmark.run_server_comparison_capture(self.config, self.output / "bad", rounds=2, seed=0, settle_seconds=0)

    def test_arm_model_overrides_the_shared_model_for_that_arm(self):
        other = self.root / "other.gguf"
        other.write_bytes(b"other")
        captured = []
        def fake_capture(**kwargs):
            captured.append(kwargs)
            return {"pair": kwargs["pair"] + 1, "mode": kwargs["side"], "position": kwargs["position"], "returncode": 0, "metrics": {"tg128_tps": 30.0}}
        self.write_config(arms=[dict(self.arms[0], model=str(other)), self.arms[1]])
        with self.patches_for_preflight(), patch(
            "bigcherry.campaign.benchmark.run_server_arm_capture", side_effect=fake_capture,
        ):
            self.assertEqual(benchmark.run_server_comparison_capture(self.config, self.output, rounds=2, seed=0, settle_seconds=0), 0)
        by_side = {item["side"]: item["model"] for item in captured}
        self.assertEqual(by_side[self.arms[0]["name"]], other.resolve())
        self.assertEqual(by_side[self.arms[1]["name"]], self.model.resolve())
        summary = json.loads((self.output / "run.json").read_text())
        self.assertEqual(list(summary["arm_model_sha256"]), [self.arms[0]["name"]])

    def quality_block(self, **updates):
        ref = self.root / "ref"
        ref.mkdir(exist_ok=True)
        (ref / "logits.bin").write_bytes(b"x")
        return {"reference_dir": str(ref), "args": ["--tp", "1"], "thresholds": {
            "max_kld_mean_increase": 0.002, "max_kld_median_increase": 0.001, "max_kld_p99_increase": 0.02,
            "max_kld_p999_increase": 0.1, "max_top1_drop_points": 0.5, "max_ppl_ratio_increase": 0.005}, **updates}

    def test_quality_block_needs_an_engine_with_a_quality_mode(self):
        self.write_config(quality=self.quality_block())
        with self.patches_for_preflight(), self.assertRaisesRegex(ValueError, "no quality measurement mode"):
            benchmark.run_server_comparison_capture(self.config, self.output, rounds=2, seed=0, settle_seconds=0)

    def test_quality_only_comparison_records_gated_arms_without_serving(self):
        from bigcherry.engine_adapters import _ADAPTERS
        import sys
        from bigcherry.engine_adapters.base import QualityCommand
        from bigcherry.engine_adapters.radiance import RadianceAdapter

        class FakeQualityAdapter(RadianceAdapter):
            def quality_command(self, kind, arm, env, run_dir, binary, model, quality):
                result = run_dir / f"{arm['name']}.kld.json"
                program = ("import json,sys;json.dump({'all':{'kld':{'mean':0.003,'median':0.001,'p99':0.02,'p99_9':0.1},"
                           "'top1_agreement':0.968,'ppl':{'candidate':8.55,'reference':8.56,'ratio':0.9988}}},open(sys.argv[1],'w'))")
                return QualityCommand((sys.executable, "-c", program, str(result)), dict(env),
                                      run_dir / f"{arm['name']}.score.log", result, 60.0)
        self.write_config(engine="radiance", execution_evidence="observe", quality_only=True, quality=self.quality_block(), server_args=[],
                          arms=[dict(arm, mode="stock") for arm in self.arms])
        with self.patches_for_preflight(), patch.dict(_ADAPTERS, {"radiance": FakeQualityAdapter()}), patch(
            "bigcherry.campaign.benchmark.run_server_arm_capture") as serve:
            self.assertEqual(benchmark.run_server_comparison_capture(self.config, self.output, rounds=2, seed=0, settle_seconds=0), 0)
        serve.assert_not_called()
        summary = json.loads((self.output / "run.json").read_text())
        self.assertEqual(summary["runs"], [])
        self.assertTrue(summary["quality"]["passed"])
        self.assertEqual(sorted(summary["quality"]["arms"]), sorted(item["name"] for item in self.arms))

    def test_ambient_patch_and_allreduce_controls_do_not_leak_into_arms(self):
        captured = []
        def fake_capture(**kwargs):
            captured.append(kwargs)
            return {"pair": kwargs["pair"] + 1, "mode": kwargs["side"], "position": kwargs["position"], "returncode": 0, "metrics": {"tg128_tps": 30.0}}
        self.write_config(arms=[dict(self.arms[0], environment={"GGML_CUDA_AR_WIRE": "bf16"}), self.arms[1]])
        ambient = {"GGML_CUDA_ALLREDUCE": "internal", "GGML_CUDA_AR_WIRE": "q8_0",
                   "BIGCHERRY_IQ_MMVQ_VDR": "1", "BIGCHERRY_PATCH_TRACE": "1", "NCCL_ALGO": "Tree"}
        with self.patches_for_preflight(), patch.dict("os.environ", ambient), patch(
            "bigcherry.campaign.benchmark.run_server_arm_capture", side_effect=fake_capture,
        ):
            self.assertEqual(benchmark.run_server_comparison_capture(self.config, self.output, rounds=2, seed=0, settle_seconds=0), 0)
        by_side = {item["side"]: item["env"] for item in captured}
        self.assertEqual(by_side[self.arms[0]["name"]].get("GGML_CUDA_AR_WIRE"), "bf16")
        for env in by_side.values():
            for key in ("GGML_CUDA_ALLREDUCE", "BIGCHERRY_IQ_MMVQ_VDR", "BIGCHERRY_PATCH_TRACE", "NCCL_ALGO"):
                self.assertNotIn(key, env)
        self.assertNotIn("GGML_CUDA_AR_WIRE", by_side[self.arms[1]["name"]])

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
        advisories = json.loads((self.output / "advisories.json").read_text())
        self.assertIn("AB_RUN_FAILURE", [item["id"] for item in advisories["findings"]])

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

    # --- MEN11: per-arm environment/settings on an engine with component settings -------------------------------

    def radiance_config(self, control_settings, subject_settings, **updates):
        arms = [dict(self.arms[0], settings=control_settings), dict(self.arms[1], mode="stock", settings=subject_settings)]
        arms[0]["mode"] = "stock"
        self.write_config(engine="radiance", arms=arms, **{"execution_evidence": "observe", **updates})

    def run_radiance(self, fake_capture=None, name="capture"):
        captured = []

        def effect(**kwargs):
            captured.append(kwargs)
            if fake_capture is not None:
                return fake_capture(**kwargs)
            return {"pair": kwargs["pair"] + 1, "mode": kwargs["side"], "position": kwargs["position"],
                    "returncode": 0, "metrics": {"tg128_tps": 30.0}}

        with self.patches_for_preflight(), patch(
            "bigcherry.campaign.benchmark.run_server_arm_capture", side_effect=effect,
        ):
            rc = benchmark.run_server_comparison_capture(self.config, self.output / name, rounds=2, seed=0, settle_seconds=0)
        return rc, captured

    def test_arm_settings_become_server_variables_and_are_recorded(self):
        self.radiance_config({"R3_MOE_WORKER_ROWS": "0"}, {"R3_MOE_WORKER_ROWS": "1"})
        rc, captured = self.run_radiance()
        self.assertEqual(rc, 0)
        by_side = {item["side"]: item for item in captured}
        stock, native = self.arms[0]["name"], self.arms[1]["name"]
        self.assertEqual(by_side[stock]["env"]["R3_MOE_WORKER_ROWS"], "0")
        self.assertEqual(by_side[native]["env"]["R3_MOE_WORKER_ROWS"], "1")
        self.assertEqual(by_side[native]["settings"], {"R3_MOE_WORKER_ROWS": "1"})
        self.assertEqual(by_side[native]["env"]["HIP_VISIBLE_DEVICES"], "0,1")
        summary = json.loads((self.output / "capture" / "run.json").read_text())
        self.assertEqual(summary["arm_variables"][native], {"environment": {}, "settings": {"R3_MOE_WORKER_ROWS": "1"}})

    def test_unknown_setting_and_bad_arm_variables_are_refused(self):
        self.radiance_config({}, {"NOT_A_SETTING": "1"})
        with self.patches_for_preflight(), self.assertRaisesRegex(ValueError, "unknown radiance component setting"):
            benchmark.run_server_comparison_capture(self.config, self.output / "a", rounds=2, seed=0, settle_seconds=0)
        self.radiance_config({}, {"R3_MOE_WORKER_ROWS": "1"})
        config = json.loads(self.config.read_text())
        config["arms"][1]["environment"] = {"ROCR_VISIBLE_DEVICES": "0"}
        self.config.write_text(json.dumps(config), encoding="utf-8")
        with self.patches_for_preflight(), self.assertRaisesRegex(ValueError, "topology belongs"):
            benchmark.run_server_comparison_capture(self.config, self.output / "b", rounds=2, seed=0, settle_seconds=0)
        config["arms"][1]["environment"] = {"R3_MOE_WORKER_ROWS": "1"}
        self.config.write_text(json.dumps(config), encoding="utf-8")
        with self.patches_for_preflight(), self.assertRaisesRegex(ValueError, "one place"):
            benchmark.run_server_comparison_capture(self.config, self.output / "c", rounds=2, seed=0, settle_seconds=0)

    def test_arm_environment_merges_over_shared_environment_for_radiance(self):
        self.radiance_config({}, {})
        config = json.loads(self.config.read_text())
        config["environment"]["R3_SHARED"] = "x"
        config["arms"][1]["environment"] = {"R3_AR_TIMING_EXTRA": "7"}
        self.config.write_text(json.dumps(config), encoding="utf-8")
        _, captured = self.run_radiance()
        by_side = {item["side"]: item["env"] for item in captured}
        self.assertEqual(by_side[self.arms[0]["name"]]["R3_SHARED"], "x")
        self.assertEqual(by_side[self.arms[1]["name"]]["R3_SHARED"], "x")
        self.assertEqual(by_side[self.arms[1]["name"]]["R3_AR_TIMING_EXTRA"], "7")
        self.assertNotIn("R3_AR_TIMING_EXTRA", by_side[self.arms[0]["name"]])

    def test_non_dispatch_engine_refuses_dispatch_modes_and_required_attestation(self):
        self.radiance_config({}, {})
        config = json.loads(self.config.read_text())
        config["arms"][1]["mode"] = "native"
        self.config.write_text(json.dumps(config), encoding="utf-8")
        with self.patches_for_preflight(), self.assertRaisesRegex(ValueError, "unsupported dispatch mode"):
            benchmark.run_server_comparison_capture(self.config, self.output / "m", rounds=2, seed=0, settle_seconds=0)
        self.radiance_config({}, {}, execution_evidence="required")
        with self.patches_for_preflight(), self.assertRaisesRegex(ValueError, "not parsed"):
            benchmark.run_server_comparison_capture(self.config, self.output / "e", rounds=2, seed=0, settle_seconds=0)

    def test_setting_activation_is_recorded_from_server_and_sidecar_logs(self):
        self.radiance_config({}, {"R3_MOE_WORKER": "1", "R3_MOE_WORKER_DEVICES": "0"})
        marker = "r3_moe_worker: runtime map owns 12 experts, 3.500 GiB\n"

        def fake(**kwargs):
            cell = kwargs["output"] / f"pair-{kwargs['pair'] + 1:03d}-{kwargs['side']}"
            cell.mkdir(parents=True)
            (cell / "server.log").write_text("server up\n",encoding="utf-8")
            sidecars = []
            if kwargs["settings"]:
                (cell / "helper0.log").write_text(marker, encoding="utf-8")
                sidecars.append({"name": "helper0", "log": str(cell / "helper0.log")})
            return {"pair": kwargs["pair"] + 1, "mode": kwargs["side"], "position": kwargs["position"],
                    "returncode": 0, "metrics": {"tg128_tps": 30.0}, "server_log": str(cell / "server.log"),
                    "sidecars": sidecars}

        rc, _ = self.run_radiance(fake)
        self.assertEqual(rc, 0)
        summary = json.loads((self.output / "capture" / "run.json").read_text())
        evidence = summary["setting_activation"][self.arms[1]["name"]]
        self.assertEqual(evidence["R3_MOE_WORKER"]["status"], "executed")
        self.assertEqual(evidence["R3_MOE_WORKER_DEVICES"]["status"], "unobservable")


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
        seen_envs = []
        class Server:
            def __init__(self, **kwargs):
                self.host, self.port = "127.0.0.1", 4567
                self.last_shutdown = None
                self.log_path = kwargs["log_path"]
                seen_envs.append(dict(kwargs.get("env_overrides", {})))

            def __enter__(self):
                self.log_path.write_text(startup, encoding="utf-8")
                return self

            def draft_stats(self):
                return None

            def __exit__(self, *args):
                self.last_shutdown = ShutdownResult("http", True, False, 0)

        with patch("bigcherry.tuning.server_runner.ServerRunner", Server), patch(
            "bigcherry.campaign.bench_runner.run_bench_runner_server_bench",
            return_value={"tg128_tps": 30.0},
        ) as bench:
            result = run_server_arm_capture(engine="llamacpp", settings={}, health_timeout_s=180,
                binary=Path("server"), model=Path("model"), extra_args=(),
                output=self.root, pair=0, side="native", position=0, env={},
                bench_configs="mtp-dual", runner_root=Path("runner"),
                required_metrics=("tg128_tps",), expected_execution=self.expected,
                execution_evidence=execution_evidence,
            )
        return result, bench, seen_envs

    def test_matching_attestation_allows_bench(self):
        startup = (
            "I llama_prepare_model_devices: using device ROCm0 (AMD Radeon RX 7900 XTX) (0000:01:00.0)\n"
            "I llama_prepare_model_devices: using device ROCm1 (AMD Radeon RX 7900 XTX) (0000:02:00.0)\n"
            "D load_tensors: layer 0 assigned to device ROCm0\n"
        )
        result, bench, seen_envs = self.run_cell(startup)
        bench.assert_called_once()
        self.assertEqual(result["returncode"], 0)
        self.assertEqual(result["metrics"], {"tg128_tps": 30.0})
        self.assertEqual(result["execution_attestation"]["backend"], "ROCm")
        self.assertEqual(seen_envs[0]["NCCL_DEBUG"], "INFO")
        self.assertEqual(seen_envs[0]["NCCL_DEBUG_SUBSYS"], "INIT")
        self.assertNotIn("NCCL_DEBUG", seen_envs[1])
        self.assertNotIn("NCCL_DEBUG_SUBSYS", seen_envs[1])

    def test_wrong_physical_device_blocks_bench(self):
        startup = "using device ROCm0 (AMD Radeon RX 7900 XTX) (0000:03:00.0)\n"
        result, bench, _ = self.run_cell(startup)
        bench.assert_not_called()
        self.assertEqual(result["returncode"], 1)
        self.assertNotIn("metrics", result)
        self.assertTrue(Path(result["attestation_preflight"]).is_file())

    def test_cpu_fallback_blocks_bench(self):
        result, bench, _ = self.run_cell("failed to initialize ROCm: no ROCm-capable device is detected\n")
        bench.assert_not_called()
        self.assertEqual(result["returncode"], 1)
        self.assertNotIn("metrics", result)
        self.assertTrue(Path(result["attestation_preflight"]).is_file())

    def test_missing_attestation_blocks_bench(self):
        result, bench, _ = self.run_cell("server is ready\n")
        bench.assert_not_called()
        self.assertEqual(result["returncode"], 1)
        self.assertNotIn("metrics", result)
        self.assertTrue(Path(result["attestation_preflight"]).is_file())

    def test_explicit_observation_collects_missing_evidence_without_admission(self):
        result, bench, _ = self.run_cell("server is ready\n", "observe")
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
                result, bench, _ = self.run_cell(startup, "observe")
                bench.assert_not_called()
                self.assertEqual(result["returncode"], 1)
                self.assertNotIn("metrics", result)
