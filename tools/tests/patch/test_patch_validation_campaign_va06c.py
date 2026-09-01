"""VA06 next slice: RD73's decode control lane, resource-evidence
producer, bit-identical correctness evaluator, and the full
run_rd73_contract_qualification() orchestrator + WORKLOAD_METRIC
registration. GPT scoping (session ses_1e0bd1ea53db4311): mirror RD08's
result/schema/promotion semantics; load every threshold from the real
contract, never hardcode. Hardware-free throughout -- subprocess.run and
run_rd73_mtp_server_lane are faked.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.experiment import contract as ec  # noqa: E402
from bigcherry.experiment import execution as ee  # noqa: E402
from bigcherry.patch import validation_campaign as vc  # noqa: E402


class WorkloadMetricRegistrationTests(unittest.TestCase):
    def test_mtp_verify_maps_to_mtp_wall_tps(self) -> None:
        self.assertEqual(ee.metric_for_workload("mtp_verify"), "mtp_wall_tps")


class _FakeServerRunner:
    """User redirect (2026-09-01): decode control now launches real
    ServerRunner-managed llama-server processes (not llama-bench), driven
    via the documented Brutus bench runner. Faked here for hardware-free
    testing, matching test_patch_validation_campaign_va06b.py's pattern."""

    instances: list["_FakeServerRunner"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        _FakeServerRunner.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


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
        with self.assertRaises(vc.PatchCampaignError):
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
        with self.assertRaises(vc.PatchCampaignError):
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
        with self.assertRaises(vc.PatchCampaignError):
            vc.run_bench_runner_server_bench(
                server_url="http://127.0.0.1:18080", bench_configs="tg128",
                runner_root=self.runner_root,
            )


class RunRd73DecodeControlLaneTests(unittest.TestCase):
    def setUp(self) -> None:
        _FakeServerRunner.instances = []
        self._tmp = tempfile.TemporaryDirectory()
        self.run_dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _run(self, *, control_tps, subject_tps, pairs=2):
        counters = {"control": 0, "subject": 0}

        def fake_bench_runner(*, server_url, bench_configs, repetitions=1, timeout_s=300, runner_root=None):
            arm = "control" if "18082" in server_url else "subject"
            values = control_tps if arm == "control" else subject_tps
            index = counters[arm]
            counters[arm] += 1
            return {"tg128_tps": values[index]}

        with mock.patch.object(vc, "run_bench_runner_server_bench", side_effect=fake_bench_runner):
            with mock.patch("bigcherry.tuning.server_runner.ServerRunner", _FakeServerRunner):
                return vc.run_rd73_decode_control_lane(
                    control_binary=Path("control-server"), subject_binary=Path("subject-server"),
                    model=Path("m.gguf"), run_dir=self.run_dir, pairs=pairs,
                )

    def test_returns_control_role_effect(self) -> None:
        result = self._run(control_tps=[90.0, 90.0], subject_tps=[100.0, 100.0])
        self.assertEqual(result["effect"].role, "control")
        self.assertEqual(result["effect"].metric, "tg128")
        artifact_path = self.run_dir / result["artifact"]["path"]
        self.assertTrue(artifact_path.is_file())

    def test_default_extra_flags_include_sm_tensor_and_fit_off(self) -> None:
        # This lane launches real llama-SERVER processes (unlike RD73's
        # activation/resource evidence, which reuses the MTP lane's own
        # servers) -- --fit off is required here, unlike llama-bench-based
        # lanes, which must never receive it (real hardware finding: a
        # hard argument-parse error).
        self._run(control_tps=[90.0, 90.0], subject_tps=[100.0, 100.0])
        for instance in _FakeServerRunner.instances:
            extra_args = instance.kwargs["extra_args"]
            self.assertIn("-sm", extra_args)
            self.assertEqual(extra_args[extra_args.index("-sm") + 1], "tensor")
            self.assertIn("--fit", extra_args)
            self.assertEqual(extra_args[extra_args.index("--fit") + 1], "off")


class EvaluateRd73ResourceEvidenceTests(unittest.TestCase):
    """User redirect (2026-09-01): resource evidence is now read from the
    MTP lane's own subject server log file (always
    BIGCHERRY_RD73_RESOURCE_TRACE=1) rather than a separate llama-bench
    probe."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.run_dir = Path(self._tmp.name)
        (self.run_dir / "logs").mkdir()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write_log(self, readings) -> Path:
        path = self.run_dir / "logs" / "subject.log"
        lines = "".join(f"BIGCHERRY_RD73_RESOURCE graph_cache_entries={v}\n" for v in readings)
        path.write_text(lines, encoding="utf-8")
        return path

    def test_real_contract_800_limit_passes_at_651(self) -> None:
        subject_log = self._write_log([300, 651, 400])
        result = vc.evaluate_rd73_resource_evidence(subject_log_path=subject_log, run_dir=self.run_dir)
        self.assertEqual(result["result"].subject_value, 651.0)

    def test_no_readings_fails_closed(self) -> None:
        subject_log = self._write_log([])
        with self.assertRaises(vc.PatchCampaignError):
            vc.evaluate_rd73_resource_evidence(subject_log_path=subject_log, run_dir=self.run_dir)


class EvaluateRd73MtpCorrectnessTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.run_dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _records(self, contents):
        return [{"order_index": i, "content": c} for i, c in enumerate(contents)]

    def test_exact_match_passes(self) -> None:
        result = vc.evaluate_rd73_mtp_correctness(
            control_requests=self._records(["a", "b"]), subject_requests=self._records(["a", "b"]),
            run_dir=self.run_dir,
        )
        self.assertTrue(all(r["ok"] for r in result["rows"]))

    def test_mismatch_raises(self) -> None:
        with self.assertRaises(vc.Rd73CorrectnessError):
            vc.evaluate_rd73_mtp_correctness(
                control_requests=self._records(["a", "b"]), subject_requests=self._records(["a", "X"]),
                run_dir=self.run_dir,
            )

    def test_missing_content_raises(self) -> None:
        control = [{"order_index": 0, "content": None}]
        subject = [{"order_index": 0, "content": "a"}]
        with self.assertRaises(vc.Rd73CorrectnessError):
            vc.evaluate_rd73_mtp_correctness(control_requests=control, subject_requests=subject, run_dir=self.run_dir)

    def test_unpaired_record_counts_raise(self) -> None:
        with self.assertRaises(vc.Rd73CorrectnessError):
            vc.evaluate_rd73_mtp_correctness(
                control_requests=self._records(["a", "b"]), subject_requests=self._records(["a"]),
                run_dir=self.run_dir,
            )


class RunRd73ContractQualificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._real_run = vc.subprocess.run
        self._tmp = tempfile.TemporaryDirectory()
        self.run_dir = Path(self._tmp.name)
        self.control_binary = self.run_dir / "control-bin"
        self.control_binary.write_text("", encoding="utf-8")
        self.subject_binary = self.run_dir / "subject-bin"
        self.subject_binary.write_text("", encoding="utf-8")
        self.model = self.run_dir / "model.gguf"
        self.model.write_text("", encoding="utf-8")
        self.corpus_path = self.run_dir / "corpus.jsonl"
        self.corpus_path.write_text("", encoding="utf-8")
        self.contract = ec.load_contracts(
            Path(__file__).resolve().parents[3] / "config" / "experiment-contracts.toml"
        ).contracts["RD73-STABLE-GRAPH-CACHE-KEY"]

    def tearDown(self) -> None:
        vc.subprocess.run = self._real_run
        self._tmp.cleanup()

    def _mtp_records(self, contents_control, contents_subject, wall_tps_control, wall_tps_subject):
        control_requests = [
            {"order_index": i, "content": c, "wall_tps": wt}
            for i, (c, wt) in enumerate(zip(contents_control, wall_tps_control))
        ]
        subject_requests = [
            {"order_index": i, "content": c, "wall_tps": wt}
            for i, (c, wt) in enumerate(zip(contents_subject, wall_tps_subject))
        ]
        return control_requests, subject_requests

    def _fake_mtp_lane(
        self, *, control_content, subject_content, control_tps, subject_tps,
        resource_readings=(651,), activation_marker="BIGCHERRY_PATCH_HIT patch=1233_rd73",
    ):
        # User redirect (2026-09-01): activation/resource evidence is now
        # read from the MTP lane's own server log files, so this fake
        # writes real log files with real content for
        # evaluate_rd73_activation_evidence()/evaluate_rd73_resource_evidence()
        # to read, rather than faking a separate subprocess probe.
        control_requests, subject_requests = self._mtp_records(
            control_content, subject_content, control_tps, subject_tps,
        )
        effect = ec.LaneEffect(
            role="positive", metric="mtp_wall_tps",
            geometric_effect_pct=100.0 * (sum(subject_tps) / len(subject_tps) - sum(control_tps) / len(control_tps))
            / (sum(control_tps) / len(control_tps)),
        )
        logs_dir = self.run_dir / "logs"
        logs_dir.mkdir(exist_ok=True)
        control_log_path = logs_dir / "rd73-mtp-control-server.log"
        subject_log_path = logs_dir / "rd73-mtp-subject-server.log"
        control_log_path.write_text("nothing\n", encoding="utf-8")
        resource_lines = "".join(f"BIGCHERRY_RD73_RESOURCE graph_cache_entries={v}\n" for v in resource_readings)
        subject_marker_line = f"{activation_marker}\n" if activation_marker else ""
        subject_log_path.write_text(subject_marker_line + resource_lines, encoding="utf-8")
        return {
            "effect": effect, "artifact": {"path": "artifacts/fake-mtp.json", "sha256": "x"},
            "stats": {}, "control_requests": control_requests, "subject_requests": subject_requests,
            "control_log_path": control_log_path, "subject_log_path": subject_log_path,
        }

    def _fake_decode_control(self, *, control_tps, subject_tps):
        effect = ec.LaneEffect(
            role="control", metric="tg128",
            geometric_effect_pct=100.0 * (subject_tps - control_tps) / control_tps,
        )
        return {"effect": effect, "artifact": {"path": "artifacts/fake-decode.json", "sha256": "x"}, "stats": {}}

    def _fake_resource_result(self, readings):
        result = ec.ResourceResult(
            metric="graph_cache_entries", unit="count", subject_value=float(max(readings)),
        )
        return {"result": result, "artifact": {"path": "artifacts/fake-resource.json", "sha256": "x"}, "readings": readings}

    def _run_qualification(self, mtp_result, decode_control_tps=(90.0, 100.0), resource_readings=(651,)):
        with mock.patch.object(vc, "run_rd73_mtp_server_lane", return_value=mtp_result):
            with mock.patch.object(
                vc, "run_rd73_decode_control_lane",
                return_value=self._fake_decode_control(
                    control_tps=decode_control_tps[0], subject_tps=decode_control_tps[1],
                ),
            ):
                with mock.patch.object(
                    vc, "run_rd73_resource_burst_session",
                    return_value=self._fake_resource_result(resource_readings),
                ):
                    return vc.run_rd73_contract_qualification(
                        contract=self.contract,
                        control_server_binary=self.control_binary, subject_server_binary=self.subject_binary,
                        model=self.model, marker_regex="BIGCHERRY_PATCH_HIT patch=1233_rd73",
                        corpus_path=self.corpus_path, run_dir=self.run_dir, decode_pairs=2,
                    )

    def test_all_green_qualifies(self) -> None:
        # +3.1% gain (>= 3.0 required), decode control regression handled
        # by the fake decode lane (subject 100 vs control 90 -> positive,
        # i.e. no regression), resource 651 (<= 800), exact correctness.
        mtp_result = self._fake_mtp_lane(
            control_content=["hello", "world"], subject_content=["hello", "world"],
            control_tps=[100.0, 100.0], subject_tps=[103.1, 103.1],
        )
        result = self._run_qualification(mtp_result)
        self.assertEqual(result["promotion"]["status"], "pass", result["promotion"])

    def test_gain_below_threshold_fails(self) -> None:
        mtp_result = self._fake_mtp_lane(
            control_content=["hello"], subject_content=["hello"],
            control_tps=[100.0], subject_tps=[101.9],  # +1.9%, below 3.0
        )
        result = self._run_qualification(mtp_result)
        self.assertEqual(result["promotion"]["status"], "fail")

    def test_resource_over_limit_fails(self) -> None:
        mtp_result = self._fake_mtp_lane(
            control_content=["hello"], subject_content=["hello"],
            control_tps=[100.0], subject_tps=[103.1],
        )
        result = self._run_qualification(mtp_result, resource_readings=(801,))
        self.assertEqual(result["promotion"]["status"], "fail")
        self.assertFalse(result["resource_gate"]["passed"])

    def test_correctness_mismatch_fails(self) -> None:
        mtp_result = self._fake_mtp_lane(
            control_content=["hello"], subject_content=["goodbye"],
            control_tps=[100.0], subject_tps=[103.1],
        )
        result = self._run_qualification(mtp_result)
        self.assertEqual(result["promotion"]["status"], "fail")
        self.assertFalse(result["correctness_gate"]["passed"])

    def test_missing_activation_evidence_invalidates(self) -> None:
        mtp_result = self._fake_mtp_lane(
            control_content=["hello"], subject_content=["hello"],
            control_tps=[100.0], subject_tps=[103.1], activation_marker=None,
        )
        result = self._run_qualification(mtp_result)
        self.assertEqual(result["promotion"]["status"], "invalid")

    def test_no_cv_gate_required(self) -> None:
        # Passing case above has no CV concept anywhere in its inputs or
        # in the real contract -- promotion succeeding without one
        # confirms no hidden CV requirement crept in.
        mtp_result = self._fake_mtp_lane(
            control_content=["hello"], subject_content=["hello"],
            control_tps=[100.0], subject_tps=[103.1],
        )
        result = self._run_qualification(mtp_result)
        self.assertNotIn("cv", json.dumps(result["promotion"]).lower())
        self.assertEqual(result["promotion"]["status"], "pass")


if __name__ == "__main__":
    unittest.main()
