"""VA14-B tests: validation-subject build parity assertion and RD08's real
positive/control lane execution wiring (tools/bigcherry/patch/
validation_campaign.py), hardware-free via a fake experiment-execution
runner -- consistent with VA14's established pattern
(test_experiment_execution_va14.py).
"""

from __future__ import annotations

import json
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.experiment import contract as ex_contract  # noqa: E402
from bigcherry.patch import validation_campaign as vc  # noqa: E402


@dataclass
class _FakeBuildEvidence:
    effective_configure: dict[str, str]
    effective_build_id: str


class AssertValidationSubjectParityTests(unittest.TestCase):
    def test_matching_configure_and_build_id_passes(self) -> None:
        control = _FakeBuildEvidence({"CMAKE_BUILD_TYPE": "Release"}, "abc123")
        subject = _FakeBuildEvidence({"CMAKE_BUILD_TYPE": "Release"}, "abc123")
        vc.assert_validation_subject_parity(control, subject, patch_id="p")  # no raise

    def test_mismatched_configure_raises(self) -> None:
        control = _FakeBuildEvidence({"CMAKE_BUILD_TYPE": "Release"}, "abc123")
        subject = _FakeBuildEvidence({"CMAKE_BUILD_TYPE": "Debug"}, "def456")
        with self.assertRaises(vc.PatchCampaignError):
            vc.assert_validation_subject_parity(control, subject, patch_id="p")

    def test_matching_configure_but_mismatched_build_id_raises(self) -> None:
        # Same requested options, but effective_build_id differs (e.g. a
        # nondeterministic environment fingerprint) -- must still fail
        # closed rather than trust configure equality alone.
        control = _FakeBuildEvidence({"CMAKE_BUILD_TYPE": "Release"}, "abc123")
        subject = _FakeBuildEvidence({"CMAKE_BUILD_TYPE": "Release"}, "zzz999")
        with self.assertRaises(vc.PatchCampaignError):
            vc.assert_validation_subject_parity(control, subject, patch_id="p")


class Rd08ValidationLaneCommandsTests(unittest.TestCase):
    def test_decode_workload_uses_tg128_shape_flags(self) -> None:
        control_cmd, subject_cmd = vc.rd08_validation_lane_commands(
            control_binary=Path("control_bin"), subject_binary=Path("subject_bin"),
            model=Path("m.gguf"), workload="decode",
        )
        self.assertIn("-p", control_cmd)
        self.assertEqual(control_cmd[control_cmd.index("-p") + 1], "0")
        self.assertEqual(control_cmd[control_cmd.index("-n") + 1], "128")
        self.assertEqual(control_cmd[0], "control_bin")
        self.assertEqual(subject_cmd[0], "subject_bin")

    def test_prefill_workload_uses_pp512_shape_flags(self) -> None:
        control_cmd, _ = vc.rd08_validation_lane_commands(
            control_binary=Path("control_bin"), subject_binary=Path("subject_bin"),
            model=Path("m.gguf"), workload="prefill",
        )
        self.assertEqual(control_cmd[control_cmd.index("-p") + 1], "512")
        self.assertEqual(control_cmd[control_cmd.index("-n") + 1], "0")

    def test_control_and_subject_commands_differ_only_by_binary(self) -> None:
        control_cmd, subject_cmd = vc.rd08_validation_lane_commands(
            control_binary=Path("control_bin"), subject_binary=Path("subject_bin"),
            model=Path("m.gguf"), workload="decode",
        )
        self.assertEqual(control_cmd[1:], subject_cmd[1:])

    def test_unmapped_workload_raises(self) -> None:
        with self.assertRaises(vc.PatchCampaignError):
            vc.rd08_validation_lane_commands(
                control_binary=Path("c"), subject_binary=Path("s"),
                model=Path("m"), workload="mtp_verify",
            )

    def test_default_extra_flags_is_empty_rd08_unaffected(self) -> None:
        control_cmd, _ = vc.rd08_validation_lane_commands(
            control_binary=Path("control_bin"), subject_binary=Path("subject_bin"),
            model=Path("m.gguf"), workload="decode",
        )
        self.assertNotIn("-sm", control_cmd)

    def test_extra_flags_appended_after_workload_flags(self) -> None:
        # VA06: RD73's decode control lane needs -sm tensor for its 27B
        # model on 2x gfx1100.
        control_cmd, subject_cmd = vc.rd08_validation_lane_commands(
            control_binary=Path("control_bin"), subject_binary=Path("subject_bin"),
            model=Path("m.gguf"), workload="decode", extra_flags=("-sm", "tensor"),
        )
        self.assertIn("-sm", control_cmd)
        self.assertEqual(control_cmd[control_cmd.index("-sm") + 1], "tensor")
        self.assertIn("-sm", subject_cmd)


class RunRd08ValidationLanesTests(unittest.TestCase):
    def test_persists_evidence_and_returns_real_lane_effects(self) -> None:
        contracts = ex_contract.load_contracts(
            Path(__file__).resolve().parents[3] / "config" / "experiment-contracts.toml"
        )
        contract = contracts.contracts["RD08-Q6K-MMVQ-VDR2"]

        values = {
            "control_bin": {"tg128": 100.0, "pp512": 1000.0},
            "subject_bin": {"tg128": 100.5, "pp512": 999.0},
        }

        def fake_run(command, capture_output, text, check, env):  # noqa: ANN001
            binary = command[0]
            metric = "tg128" if "-n" in command and command[command.index("-n") + 1] == "128" else "pp512"
            value = values[binary][metric]

            class _Result:
                returncode = 0
                stdout = f"ggml_cuda_init: found 1 ROCm devices\n{metric} | {value} t/s\n"
                stderr = ""

            return _Result()

        real_subprocess_run = vc.subprocess.run
        vc.subprocess.run = fake_run
        try:
            result = vc.run_rd08_validation_lanes(
                contract=contract,
                control_binary=Path("control_bin"), subject_binary=Path("subject_bin"),
                model=Path("m.gguf"), model_ref=contract.positive.models[0],
                hip_path=Path("H:/hip"), run_dir=Path(self._tmp_dir()),
                control_build_identity={"effective_build_id": "c1"},
                subject_build_identity={"effective_build_id": "s1"},
                pairs=2,
            )
        finally:
            vc.subprocess.run = real_subprocess_run

        self.assertEqual(len(result["effects"]), 2)
        roles = {effect.role for effect in result["effects"]}
        self.assertEqual(roles, {"positive", "control"})
        artifact_path = Path(self._tmp_dir()) / result["artifact"]["path"]
        self.assertTrue(artifact_path.exists())
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["contract_id"], "RD08-Q6K-MMVQ-VDR2")
        self.assertEqual(payload["model_ref"], contract.positive.models[0])
        self.assertGreater(len(payload["raw_logs"]), 0)
        self.assertIn("stdout", payload["raw_logs"][0])
        self.assertEqual(
            payload["validation_build_identities"]["control"]["effective_build_id"], "c1",
        )
        self.assertEqual(payload["lanes"]["positive"]["metric"], "tg128")
        self.assertEqual(payload["lanes"]["control"]["metric"], "pp512")

    def test_lane_env_is_sanitized_of_inherited_dispatch_overrides(self) -> None:
        contracts = ex_contract.load_contracts(
            Path(__file__).resolve().parents[3] / "config" / "experiment-contracts.toml"
        )
        contract = contracts.contracts["RD08-Q6K-MMVQ-VDR2"]
        seen_envs: list[dict[str, str]] = []

        def fake_run(command, capture_output, text, check, env):  # noqa: ANN001
            seen_envs.append(env)

            class _Result:
                returncode = 0
                stdout = "ggml_cuda_init: found 1 ROCm devices\ntg128 | 100.0 t/s\npp512 | 100.0 t/s\n"
                stderr = ""

            return _Result()

        real_subprocess_run = vc.subprocess.run
        real_hip_env = vc._hip_env
        vc.subprocess.run = fake_run
        vc._hip_env = lambda hip_path: {
            "PATH": "x", "GGML_HIP_DISPATCH_MODE": "replay", "GGML_HIP_FORCE_KERNEL": "1",
            "BIGCHERRY_DEBUG": "1", "GGML_CUDA_DISABLE_FUSION": "1",
        }
        try:
            vc.run_rd08_validation_lanes(
                contract=contract, control_binary=Path("control_bin"), subject_binary=Path("subject_bin"),
                model=Path("m.gguf"), model_ref=contract.positive.models[0],
                hip_path=Path("H:/hip"), run_dir=Path(self._tmp_dir()),
                control_build_identity={}, subject_build_identity={}, pairs=1,
            )
        finally:
            vc.subprocess.run = real_subprocess_run
            vc._hip_env = real_hip_env

        self.assertGreater(len(seen_envs), 0)
        for env in seen_envs:
            self.assertNotIn("GGML_HIP_DISPATCH_MODE", env)
            self.assertNotIn("GGML_HIP_FORCE_KERNEL", env)
            self.assertNotIn("BIGCHERRY_DEBUG", env)
            self.assertNotIn("GGML_CUDA_DISABLE_FUSION", env)

    def test_validation_lanes_artifact_is_bound_into_artifact_refs(self) -> None:
        import hashlib
        import tempfile
        from bigcherry.patch import evidence as patch_evidence

        run_dir = Path(tempfile.mkdtemp())
        artifact_ref = vc._write_bound_artifact(run_dir, "validation-lanes.json", {"k": "v"})
        refs = patch_evidence._artifact_refs(run_dir)
        matching = [r for r in refs if r["path"] == "artifacts/validation-lanes.json"]
        self.assertEqual(len(matching), 1)
        self.assertEqual(matching[0]["sha256"], artifact_ref["sha256"])
        expected_hash = hashlib.sha256((run_dir / "artifacts" / "validation-lanes.json").read_bytes()).hexdigest()
        self.assertEqual(matching[0]["sha256"], expected_hash)

    def _tmp_dir(self) -> str:
        if not hasattr(self, "_tmp"):
            import tempfile
            self._tmp = tempfile.mkdtemp()
        return self._tmp


if __name__ == "__main__":
    unittest.main()
