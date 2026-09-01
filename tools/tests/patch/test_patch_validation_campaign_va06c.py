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


def _fake_subprocess_run_factory(*, resource_readings=None, activation_marker=None):
    """One fake covering the three subprocess-based RD73 probes
    (activation, decode control, resource) -- differentiated by env vars,
    matching how the real probes are actually distinguished."""
    resource_readings = resource_readings or []

    def fake_run(command, **kwargs):
        env = kwargs.get("env") or {}

        class _Result:
            pass

        result = _Result()
        result.returncode = 0
        base = "ggml_cuda_init: found 1 ROCm devices\n"
        if env.get("BIGCHERRY_RD73_RESOURCE_TRACE") == "1":
            lines = "".join(f"BIGCHERRY_RD73_RESOURCE graph_cache_entries={v}\n" for v in resource_readings)
            result.stdout = base + lines
            result.stderr = ""
        elif env.get("BIGCHERRY_PATCH_TRACE") == "1":
            marker_line = f"{activation_marker}\n" if activation_marker and "subject" in command[0] else ""
            result.stdout = base + marker_line
            result.stderr = ""
        else:
            # decode control lane (paired llama-bench, no BIGCHERRY_* env)
            value = 100.0 if "subject" in command[0] else 90.0
            result.stdout = f"{base}tg128 | {value} t/s\n"
            result.stderr = ""
        return result

    return fake_run


class RunRd73DecodeControlLaneTests(unittest.TestCase):
    def setUp(self) -> None:
        self._real_run = vc.subprocess.run
        self._tmp = tempfile.TemporaryDirectory()
        self.run_dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        vc.subprocess.run = self._real_run
        self._tmp.cleanup()

    def test_returns_control_role_effect(self) -> None:
        vc.subprocess.run = _fake_subprocess_run_factory()
        result = vc.run_rd73_decode_control_lane(
            control_binary=Path("control_bin"), subject_binary=Path("subject_bin"),
            model=Path("m.gguf"), hip_path=Path("H:/hip"), run_dir=self.run_dir, pairs=2,
        )
        self.assertEqual(result["effect"].role, "control")
        self.assertEqual(result["effect"].metric, "tg128")
        artifact_path = self.run_dir / result["artifact"]["path"]
        self.assertTrue(artifact_path.is_file())

    def test_default_extra_flags_include_sm_tensor(self) -> None:
        # RD73's real contract model (tierL-qwen27b-q8) needs -sm tensor
        # on Brutus's dual gfx1100 GPUs -- the default -sm layer split
        # understates throughput by roughly 2-10x.
        seen_commands = []

        def fake_run(command, **kwargs):
            seen_commands.append(command)
            class _Result:
                returncode = 0
                stdout = "ggml_cuda_init: found 1 ROCm devices\ntg128 | 100.0 t/s\n"
                stderr = ""
            return _Result()

        vc.subprocess.run = fake_run
        vc.run_rd73_decode_control_lane(
            control_binary=Path("control_bin"), subject_binary=Path("subject_bin"),
            model=Path("m.gguf"), hip_path=Path("H:/hip"), run_dir=self.run_dir, pairs=1,
        )
        for command in seen_commands:
            self.assertIn("-sm", command)
            self.assertEqual(command[command.index("-sm") + 1], "tensor")


class RunRd73ResourceEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._real_run = vc.subprocess.run
        self._tmp = tempfile.TemporaryDirectory()
        self.run_dir = Path(self._tmp.name)
        self.subject_binary = self.run_dir / "subject-bin"
        self.subject_binary.write_text("", encoding="utf-8")
        self.model = self.run_dir / "model.gguf"
        self.model.write_text("", encoding="utf-8")

    def tearDown(self) -> None:
        vc.subprocess.run = self._real_run
        self._tmp.cleanup()

    def test_real_contract_800_limit_passes_at_651(self) -> None:
        vc.subprocess.run = _fake_subprocess_run_factory(resource_readings=[300, 651, 400])
        result = vc.run_rd73_resource_evidence(
            subject_binary=self.subject_binary, model=self.model,
            hip_path=Path("H:/hip"), workdir=self.run_dir, run_dir=self.run_dir,
        )
        self.assertEqual(result["result"].subject_value, 651.0)

    def test_no_readings_fails_closed(self) -> None:
        vc.subprocess.run = _fake_subprocess_run_factory(resource_readings=[])
        with self.assertRaises(vc.PatchCampaignError):
            vc.run_rd73_resource_evidence(
                subject_binary=self.subject_binary, model=self.model,
                hip_path=Path("H:/hip"), workdir=self.run_dir, run_dir=self.run_dir,
            )


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

    def _fake_mtp_lane(self, *, control_content, subject_content, control_tps, subject_tps):
        control_requests, subject_requests = self._mtp_records(
            control_content, subject_content, control_tps, subject_tps,
        )
        effect = ec.LaneEffect(
            role="positive", metric="mtp_wall_tps",
            geometric_effect_pct=100.0 * (sum(subject_tps) / len(subject_tps) - sum(control_tps) / len(control_tps))
            / (sum(control_tps) / len(control_tps)),
        )
        return {
            "effect": effect, "artifact": {"path": "artifacts/fake-mtp.json", "sha256": "x"},
            "stats": {}, "control_requests": control_requests, "subject_requests": subject_requests,
        }

    def _run_qualification(self, mtp_result, resource_readings, activation_marker):
        vc.subprocess.run = _fake_subprocess_run_factory(
            resource_readings=resource_readings, activation_marker=activation_marker,
        )
        with mock.patch.object(vc, "run_rd73_mtp_server_lane", return_value=mtp_result):
            return vc.run_rd73_contract_qualification(
                contract=self.contract, control_binary=self.control_binary,
                subject_binary=self.subject_binary,
                control_server_binary=self.control_binary, subject_server_binary=self.subject_binary,
                model=self.model, marker_regex="BIGCHERRY_PATCH_HIT patch=1233_rd73",
                corpus_path=self.corpus_path, hip_path=Path("H:/hip"),
                workdir=self.run_dir, run_dir=self.run_dir, decode_pairs=2,
            )

    def test_all_green_qualifies(self) -> None:
        # +3.1% gain (>= 3.0 required), decode control regression handled
        # by the fake decode lane (subject 100 vs control 90 -> positive,
        # i.e. no regression), resource 651 (<= 800), exact correctness.
        mtp_result = self._fake_mtp_lane(
            control_content=["hello", "world"], subject_content=["hello", "world"],
            control_tps=[100.0, 100.0], subject_tps=[103.1, 103.1],
        )
        result = self._run_qualification(
            mtp_result, resource_readings=[651], activation_marker="BIGCHERRY_PATCH_HIT patch=1233_rd73",
        )
        self.assertEqual(result["promotion"]["status"], "pass", result["promotion"])

    def test_gain_below_threshold_fails(self) -> None:
        mtp_result = self._fake_mtp_lane(
            control_content=["hello"], subject_content=["hello"],
            control_tps=[100.0], subject_tps=[101.9],  # +1.9%, below 3.0
        )
        result = self._run_qualification(
            mtp_result, resource_readings=[651], activation_marker="BIGCHERRY_PATCH_HIT patch=1233_rd73",
        )
        self.assertEqual(result["promotion"]["status"], "fail")

    def test_resource_over_limit_fails(self) -> None:
        mtp_result = self._fake_mtp_lane(
            control_content=["hello"], subject_content=["hello"],
            control_tps=[100.0], subject_tps=[103.1],
        )
        result = self._run_qualification(
            mtp_result, resource_readings=[801], activation_marker="BIGCHERRY_PATCH_HIT patch=1233_rd73",
        )
        self.assertEqual(result["promotion"]["status"], "fail")
        self.assertFalse(result["resource_gate"]["passed"])

    def test_correctness_mismatch_fails(self) -> None:
        mtp_result = self._fake_mtp_lane(
            control_content=["hello"], subject_content=["goodbye"],
            control_tps=[100.0], subject_tps=[103.1],
        )
        result = self._run_qualification(
            mtp_result, resource_readings=[651], activation_marker="BIGCHERRY_PATCH_HIT patch=1233_rd73",
        )
        self.assertEqual(result["promotion"]["status"], "fail")
        self.assertFalse(result["correctness_gate"]["passed"])

    def test_missing_activation_evidence_invalidates(self) -> None:
        mtp_result = self._fake_mtp_lane(
            control_content=["hello"], subject_content=["hello"],
            control_tps=[100.0], subject_tps=[103.1],
        )
        result = self._run_qualification(mtp_result, resource_readings=[651], activation_marker=None)
        self.assertEqual(result["promotion"]["status"], "invalid")

    def test_no_cv_gate_required(self) -> None:
        # Passing case above has no CV concept anywhere in its inputs or
        # in the real contract -- promotion succeeding without one
        # confirms no hidden CV requirement crept in.
        mtp_result = self._fake_mtp_lane(
            control_content=["hello"], subject_content=["hello"],
            control_tps=[100.0], subject_tps=[103.1],
        )
        result = self._run_qualification(
            mtp_result, resource_readings=[651], activation_marker="BIGCHERRY_PATCH_HIT patch=1233_rd73",
        )
        self.assertNotIn("cv", json.dumps(result["promotion"]).lower())
        self.assertEqual(result["promotion"]["status"], "pass")


if __name__ == "__main__":
    unittest.main()
