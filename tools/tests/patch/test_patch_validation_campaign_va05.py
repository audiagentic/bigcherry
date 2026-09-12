"""VA05 hardware-free tests: run_rd58_state_restore_evidence() -- an
RD58-scoped validation-domain state-restore correctness/activation/
controls producer, analogous to RD04/RD08's producers but for a
correctness-only contract with no performance claim. Hardware-free via
an injected fake subprocess.run().
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.patch import validation_campaign as vc  # noqa: E402


class _Result:
    def __init__(self, returncode: int, stdout: str, stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class RunRd58StateRestoreEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._real_subprocess_run = vc.subprocess.run

    def tearDown(self) -> None:
        vc.subprocess.run = self._real_subprocess_run

    def _run_dir(self) -> Path:
        return Path(tempfile.mkdtemp())

    def test_subject_hit_control_miss_with_all_passing(self) -> None:
        def fake_run(command, capture_output, text, check, env):  # noqa: ANN001
            self.assertEqual(env.get("GGML_CUDA_REGISTER_HOST"), "1")
            self.assertIn("-sm", command)
            self.assertEqual(command[command.index("-sm") + 1], "tensor")
            binary = command[0]
            if binary == "subject_bin":
                return _Result(0, "pinned state buffer (55966349 bytes) for restore\nAll tests passed.\n")
            return _Result(0, "All tests passed.\n")

        vc.subprocess.run = fake_run
        run_dir = self._run_dir()
        result = vc.run_rd58_state_restore_evidence(
            control_binary=Path("control_bin"), subject_binary=Path("subject_bin"),
            model=Path("m.gguf"), hip_path=Path("H:/hip"), run_dir=run_dir,
            campaign_id="campaign123",
            control_build_identity={"effective_build_id": "c1"},
            subject_build_identity={"effective_build_id": "s1"},
            repetitions=2,
        )
        self.assertTrue(result["correctness_passed"])
        self.assertTrue(result["subject_hit"])
        self.assertFalse(result["control_hit"])
        self.assertTrue(result["controls_passed"])

        correctness_path = run_dir / "rd58-correctness.json"
        self.assertTrue(correctness_path.is_file())
        doc = json.loads(correctness_path.read_text(encoding="utf-8"))
        self.assertEqual(doc["ops"], ["STATE_RESTORE_SEQ_CP_HOST"])
        self.assertTrue(doc["passed"])

        import hashlib
        self.assertEqual(
            result["correctness_artifact"]["sha256"],
            hashlib.sha256(correctness_path.read_bytes()).hexdigest(),
        )

    def test_control_hit_is_detected_as_a_contamination_signal(self) -> None:
        def fake_run(command, capture_output, text, check, env):  # noqa: ANN001
            return _Result(0, "pinned state buffer (12345 bytes) for restore\nAll tests passed.\n")

        vc.subprocess.run = fake_run
        result = vc.run_rd58_state_restore_evidence(
            control_binary=Path("control_bin"), subject_binary=Path("subject_bin"),
            model=Path("m.gguf"), hip_path=Path("H:/hip"), run_dir=self._run_dir(),
            campaign_id="campaign123", control_build_identity={}, subject_build_identity={},
            repetitions=1,
        )
        self.assertTrue(result["subject_hit"])
        self.assertTrue(result["control_hit"])

    def test_subject_nonzero_returncode_fails_correctness(self) -> None:
        def fake_run(command, capture_output, text, check, env):  # noqa: ANN001
            if command[0] == "subject_bin":
                return _Result(1, "", "Test 4: seq copy (host) failed\n")
            return _Result(0, "All tests passed.\n")

        vc.subprocess.run = fake_run
        result = vc.run_rd58_state_restore_evidence(
            control_binary=Path("control_bin"), subject_binary=Path("subject_bin"),
            model=Path("m.gguf"), hip_path=Path("H:/hip"), run_dir=self._run_dir(),
            campaign_id="campaign123", control_build_identity={}, subject_build_identity={},
            repetitions=2,
        )
        self.assertFalse(result["correctness_passed"])

    def test_control_nonzero_returncode_fails_controls_not_correctness(self) -> None:
        def fake_run(command, capture_output, text, check, env):  # noqa: ANN001
            if command[0] == "control_bin":
                return _Result(1, "", "crash\n")
            return _Result(0, "All tests passed.\n")

        vc.subprocess.run = fake_run
        result = vc.run_rd58_state_restore_evidence(
            control_binary=Path("control_bin"), subject_binary=Path("subject_bin"),
            model=Path("m.gguf"), hip_path=Path("H:/hip"), run_dir=self._run_dir(),
            campaign_id="campaign123", control_build_identity={}, subject_build_identity={},
            repetitions=2,
        )
        self.assertTrue(result["correctness_passed"])
        self.assertFalse(result["controls_passed"])

    def test_subject_failure_also_fails_controls(self) -> None:
        # GPT round 3: the controls artifact claims no crash/regression
        # across BOTH control and subject repeated execution -- a subject
        # that crashes on repeated runs must fail controls too, not just
        # correctness.
        def fake_run(command, capture_output, text, check, env):  # noqa: ANN001
            if command[0] == "subject_bin":
                return _Result(1, "", "crash\n")
            return _Result(0, "All tests passed.\n")

        vc.subprocess.run = fake_run
        result = vc.run_rd58_state_restore_evidence(
            control_binary=Path("control_bin"), subject_binary=Path("subject_bin"),
            model=Path("m.gguf"), hip_path=Path("H:/hip"), run_dir=self._run_dir(),
            campaign_id="campaign123", control_build_identity={}, subject_build_identity={},
            repetitions=2,
        )
        self.assertFalse(result["correctness_passed"])
        self.assertFalse(result["controls_passed"])

    def test_observed_devices_persisted_into_artifacts(self) -> None:
        def fake_run(command, capture_output, text, check, env):  # noqa: ANN001
            return _Result(0, "All tests passed.\n")

        vc.subprocess.run = fake_run
        run_dir = self._run_dir()
        devices = {
            "hip_visible_devices": ["0", "1"], "rocr_visible_devices": ["0", "1"], "gpu_count": 2,
        }
        vc.run_rd58_state_restore_evidence(
            control_binary=Path("control_bin"), subject_binary=Path("subject_bin"),
            model=Path("m.gguf"), hip_path=Path("H:/hip"), run_dir=run_dir,
            campaign_id="campaign123", control_build_identity={}, subject_build_identity={},
            observed_devices=devices, repetitions=1,
        )
        correctness_doc = json.loads((run_dir / "rd58-correctness.json").read_text(encoding="utf-8"))
        controls_doc = json.loads((run_dir / "performance.json").read_text(encoding="utf-8"))
        self.assertEqual(correctness_doc["hardware"], devices)
        self.assertEqual(controls_doc["hardware"], devices)

    def test_ambient_device_visibility_is_preserved(self) -> None:
        # Unlike RD04/RD08, RD58 must NOT restrict HIP_VISIBLE_DEVICES --
        # it needs 2+ real GPUs. Confirm the env passed to subprocess.run
        # inherits whatever HIP_VISIBLE_DEVICES the parent process set.
        import os
        seen_envs = []

        def fake_run(command, capture_output, text, check, env):  # noqa: ANN001
            seen_envs.append(env)
            return _Result(0, "All tests passed.\n")

        vc.subprocess.run = fake_run
        old_value = os.environ.get("HIP_VISIBLE_DEVICES")
        os.environ["HIP_VISIBLE_DEVICES"] = "0,1"
        try:
            vc.run_rd58_state_restore_evidence(
                control_binary=Path("control_bin"), subject_binary=Path("subject_bin"),
                model=Path("m.gguf"), hip_path=Path("H:/hip"), run_dir=self._run_dir(),
                campaign_id="campaign123", control_build_identity={}, subject_build_identity={},
                repetitions=1,
            )
        finally:
            if old_value is None:
                os.environ.pop("HIP_VISIBLE_DEVICES", None)
            else:
                os.environ["HIP_VISIBLE_DEVICES"] = old_value

        self.assertTrue(seen_envs)
        for env in seen_envs:
            self.assertEqual(env.get("HIP_VISIBLE_DEVICES"), "0,1")

    def test_ambient_rocr_visible_devices_is_stripped(self) -> None:
        # GPT review (req_8429aa8e0d35496e, 2026-09-11): an ambient
        # ROCR_VISIBLE_DEVICES left over from an unrelated earlier
        # command must never reach RD58's launched process -- this
        # module's selector contract is HIP-only now (PNRO17). Preserving
        # ambient HIP_VISIBLE_DEVICES (the test above) does not mean
        # preserving ROCR too; the two are independent risks.
        import os
        seen_envs = []

        def fake_run(command, capture_output, text, check, env):  # noqa: ANN001
            seen_envs.append(env)
            return _Result(0, "All tests passed.\n")

        vc.subprocess.run = fake_run
        old = dict(os.environ)
        try:
            os.environ["HIP_VISIBLE_DEVICES"] = "0,1"
            os.environ["ROCR_VISIBLE_DEVICES"] = "6"
            vc.run_rd58_state_restore_evidence(
                control_binary=Path("control_bin"), subject_binary=Path("subject_bin"),
                model=Path("m.gguf"), hip_path=Path("H:/hip"), run_dir=self._run_dir(),
                campaign_id="campaign123", control_build_identity={}, subject_build_identity={},
                repetitions=1,
            )
        finally:
            os.environ.clear()
            os.environ.update(old)

        self.assertTrue(seen_envs)
        for env in seen_envs:
            self.assertNotIn("ROCR_VISIBLE_DEVICES", env)
            self.assertEqual(env.get("HIP_VISIBLE_DEVICES"), "0,1")


class Rd58CliWiringTests(unittest.TestCase):
    def setUp(self) -> None:
        import inspect
        self.source = inspect.getsource(vc.run)

    def test_run_rd58_state_restore_flag_exists(self) -> None:
        import inspect
        main_source = inspect.getsource(vc.main)
        self.assertIn('"--run-rd58-state-restore"', main_source)

    def test_mutually_exclusive_with_rd04_and_rd08_modes(self) -> None:
        self.assertIn(
            "--run-rd58-state-restore is mutually exclusive with the", self.source
        )
        self.assertIn(
            "if args.run_rd04_benchmark or args.run_rd08_lanes or args.run_rd08_contract:",
            self.source,
        )

    def test_rd58_only_gating(self) -> None:
        self.assertIn(
            'descriptor.experiment_contract != "RD58-PIN-STATE-BUFFER-MULTIGPU-RESTORE"',
            self.source,
        )

    def test_populates_contract_promotions_from_a_real_promotion_gate_call(self) -> None:
        # PRBE109 fix (2026-09-13): RD58's real correctness gate was always
        # computed in this block, but its promotion verdict never reached
        # contract_promotions, so eligible_for_validated_state could never
        # become True for RD58 through this CLI path regardless of evidence
        # completeness -- a real bug, now fixed. This test previously
        # asserted the OLD (buggy) invariant that this block never touches
        # contract_promotions at all; it now asserts the corrected one:
        # the assignment exists, is keyed by the real contract id (not a
        # hardcoded string), and is populated from a real
        # evaluate_promotion_gate() call, not a fabricated always-pass value.
        rd58_block_start = self.source.index("if args.run_rd58_state_restore:")
        rd58_block = self.source[rd58_block_start:self.source.index("# VA06: RD73 execution")]
        self.assertIn("contract_promotions[rd58_contract_check.id] = rd58_promotion", rd58_block)
        self.assertIn("_ec.evaluate_promotion_gate(", rd58_block)
        self.assertIn("rd58_aggregated_effects", rd58_block)
        self.assertIn("rd58_trigger_proof", rd58_block)

    def test_gpu_preflight_goes_through_the_shared_visibility_primitive(self) -> None:
        # PVPS02 step 7: the inline HIP/ROCR guard (including the "0,0" not
        # counting as 2 distinct GPUs check -- GPT round 3) was replaced by
        # a call to require_device_visibility(), which carries its own
        # dedicated duplicate-id test coverage in
        # test_experiment_execution_va14.py's RequireDeviceVisibilityTests.
        # This just proves RD58 actually reaches that shared primitive
        # rather than reimplementing selector validation inline again.
        self.assertIn("_require_device_visibility(", self.source)
        self.assertIn("rd58_observed_devices = rd58_visibility.document()", self.source)

    def test_validation_context_uses_rd58_build_identities_when_rd58_ran(self) -> None:
        # GPT round 3: ValidationContext's build check must be evaluated
        # against RD58's own test-save-load-state builds, not the generic
        # llama-bench builds, when --run-rd58-state-restore ran.
        self.assertIn("rd58_control_build_evidence.effective_build_id", self.source)
        self.assertIn(
            "build_evidence=(rd58_build_evidence if args.run_rd58_state_restore else build_evidence)",
            self.source,
        )


if __name__ == "__main__":
    unittest.main()
