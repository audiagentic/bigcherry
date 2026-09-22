"""VA06 next slice: RD73's decode control lane + the shared workload-metric
registration and bench-runner parsing.

PA36 RD73 legacy compatibility retirement: the decode control lane moved
from shared validation_campaign.py (run_rd73_decode_control_lane) into
RD73's producer (patches/1233_rd73_stable_graph_cache_key/
validation/producer.py, _run_decode_control_lane). This test now exercises
the producer-side lane directly.

The resource-evidence (evaluate_rd73_resource_evidence), bit-identical
correctness (evaluate_rd73_mtp_correctness), and full-qualification
orchestrator (run_rd73_contract_qualification) shared functions were
eliminated/inlined into the producer's run() during the same retirement;
their unit tests were retired with them. The producer's run() end-to-end
coverage lives in test_patch_validation_campaign_rd73_contract_cli.py.

GPT scoping (session ses_1e0bd1ea53db4311): mirror RD08's result/schema/
promotion semantics; load every threshold from the real contract, never
hardcode. Hardware-free throughout -- subprocess.run and the bench runner
are faked.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.experiment import attestation as att  # noqa: E402
from bigcherry.experiment import contract as ec  # noqa: E402
from bigcherry.experiment import execution as ee  # noqa: E402
from bigcherry.campaign import bench_runner  # noqa: E402
from bigcherry.patch import validation_campaign as vc  # noqa: E402

PRODUCER_DIR = Path(
    "patches/1233_rd73_stable_graph_cache_key/validation"
)
PRODUCER_MODULE = "patches_1233_rd73_stable_graph_cache_key_validation_producer_va06c"


def _load_producer() -> Any:
    """Load the producer module with the required sys.modules registration."""
    spec = importlib.util.spec_from_file_location(
        PRODUCER_MODULE,
        PRODUCER_DIR / "producer.py",
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[PRODUCER_MODULE] = module
    spec.loader.exec_module(module)
    return module


class WorkloadMetricRegistrationTests(unittest.TestCase):
    def test_mtp_verify_maps_to_mtp_wall_tps(self) -> None:
        self.assertEqual(ee.metric_for_workload("mtp_verify"), "mtp_wall_tps")


class _FakeServerRunner:
    """User redirect (2026-09-01): decode control now launches real
    ServerRunner-managed llama-server processes (not llama-bench), driven
    via the documented Brutus bench runner. Faked here for hardware-free
    testing, matching test_patch_validation_campaign_va06b.py's pattern.

    VA25: exposes launch()/wait_healthy()/shutdown() rather than just the
    context-manager protocol, matching what AttestedServerSession actually
    calls -- the lane no longer uses ``with runner:`` on a raw ServerRunner
    directly."""

    instances: list["_FakeServerRunner"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.host = kwargs.get("host", "127.0.0.1")
        self.port = kwargs.get("port", 0)
        # AttestedServerSession reads this file right after launch()/
        # wait_healthy() succeed, so the fake must produce one just like
        # real ServerRunner does (stdout redirect on launch).
        log_path = kwargs.get("log_path")
        if log_path is not None:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            Path(log_path).write_text("", encoding="utf-8")
        _FakeServerRunner.instances.append(self)

    def launch(self) -> None:
        pass

    def wait_healthy(self, timeout_s: int = 180) -> None:
        pass

    def shutdown(self, timeout_s: int = 90):
        return None


class RunBenchRunnerServerBenchTests(unittest.TestCase):
    def setUp(self) -> None:
        self._real_run = vc.subprocess.run
        self._tmp = tempfile.TemporaryDirectory()
        self.runner_root = Path(self._tmp.name)
        (self.runner_root / "bench").mkdir()
        (self.runner_root / "bench" / "run_bench.py").write_text("", encoding="utf-8")

    def tearDown(self) -> None:
        vc.subprocess.run = self._real_run
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

        vc.subprocess.run = fake_run
        metrics = vc.run_bench_runner_server_bench(
            server_url="http://127.0.0.1:18080", bench_configs="tg128",
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

        vc.subprocess.run = fake_run
        metrics = vc.run_bench_runner_server_bench(
            server_url="http://127.0.0.1:18082", bench_configs="tg128",
            runner_root=self.runner_root,
        )
        self.assertEqual(metrics["tg128_tps"], 27.29)
        self.assertNotIn("tg128_stddev", metrics)

    def test_missing_runner_script_fails_closed(self) -> None:
        with self.assertRaises(bench_runner.BenchRunnerError):
            vc.run_bench_runner_server_bench(
                server_url="http://127.0.0.1:18080", bench_configs="tg128",
                runner_root=Path("/nonexistent"),
            )

    def test_nonzero_exit_fails_closed(self) -> None:
        def fake_run(command, **kwargs):
            class _Result:
                returncode = 1
                stdout = "error"
                stderr = "boom"
            return _Result()

        vc.subprocess.run = fake_run
        with self.assertRaises(bench_runner.BenchRunnerError):
            vc.run_bench_runner_server_bench(
                server_url="http://127.0.0.1:18080", bench_configs="tg128",
                runner_root=self.runner_root,
            )

    def test_no_parseable_metrics_fails_closed(self) -> None:
        def fake_run(command, **kwargs):
            class _Result:
                returncode = 0
                stdout = "nothing useful here"
                stderr = ""
            return _Result()

        vc.subprocess.run = fake_run
        with self.assertRaises(bench_runner.BenchRunnerError):
            vc.run_bench_runner_server_bench(
                server_url="http://127.0.0.1:18080", bench_configs="tg128",
                runner_root=self.runner_root,
            )


class RunRd73DecodeControlLaneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.producer = _load_producer()

    def setUp(self) -> None:
        _FakeServerRunner.instances = []
        self._tmp = tempfile.TemporaryDirectory()
        self.run_dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _make_ctx(self) -> Any:
        class _Ctx:
            pass

        ctx = _Ctx()
        ctx.model = Path("m.gguf")
        ctx.workdir = self.run_dir
        return ctx

    def _run(self, *, control_tps, subject_tps):
        counters = {"control": 0, "subject": 0}

        def fake_bench_runner(*, server_url, bench_configs, repetitions=1, env_overrides=None):
            arm = "control" if "18082" in server_url else "subject"
            values = control_tps if arm == "control" else subject_tps
            index = counters[arm]
            counters[arm] += 1
            return {"tg128_tps": values[index]}

        # The producer imports run_bench_runner_server_bench from
        # bigcherry.campaign.bench_runner at call time, so the fake must be
        # installed on the source module attribute. AttestedServerSession's
        # ServerRunner is patched on server_execution's namespace (where its
        # own `from ..tuning.server_runner import ServerRunner` already bound
        # the name). Attestation content parsing is fixed to always match;
        # it is tested in test_attested_server_session.py.
        with mock.patch.object(bench_runner, "run_bench_runner_server_bench", side_effect=fake_bench_runner):
            with mock.patch("bigcherry.experiment.server_execution.ServerRunner", _FakeServerRunner):
                with mock.patch(
                    "bigcherry.experiment.server_execution.parse_llama_server_attestation",
                    return_value=att.ExecutionAttestation(
                        backend="ROCm",
                        devices=(
                            att.ObservedDevice(architecture="gfx1100", locator=None),
                            att.ObservedDevice(architecture="gfx1100", locator=None),
                        ),
                    ),
                ):
                    return self.producer._run_decode_control_lane(
                        ctx=self._make_ctx(),
                        control_binary=Path("control-server"),
                        subject_binary=Path("subject-server"),
                        expected_execution=att.ExecutionIdentity(
                            backend="ROCm", architectures=("gfx1100", "gfx1100"),
                        ),
                        selector_env={},
                    )

    def test_returns_control_role_effect(self) -> None:
        # The producer's decode lane runs a fixed 10 measured pairs.
        result = self._run(
            control_tps=[90.0] * 10, subject_tps=[100.0] * 10,
        )
        self.assertEqual(result.role, "control")
        self.assertEqual(result.metric, "tg128")

    def test_default_extra_flags_include_sm_tensor_and_fit_off(self) -> None:
        # This lane launches real llama-SERVER processes (unlike RD73's
        # activation/resource evidence, which reuses the MTP lane's own
        # servers) -- --fit off is required here, unlike llama-bench-based
        # lanes, which must never receive it (real hardware finding: a
        # hard argument-parse error).
        self._run(control_tps=[90.0] * 10, subject_tps=[100.0] * 10)
        for instance in _FakeServerRunner.instances:
            extra_args = instance.kwargs["extra_args"]
            self.assertIn("-sm", extra_args)
            self.assertEqual(extra_args[extra_args.index("-sm") + 1], "tensor")
            self.assertIn("--fit", extra_args)
            self.assertEqual(extra_args[extra_args.index("--fit") + 1], "off")


if __name__ == "__main__":
    unittest.main()
