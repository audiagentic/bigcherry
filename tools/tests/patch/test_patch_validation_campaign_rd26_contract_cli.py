"""Real hardware-free coverage for --run-rd26-contract's CLI wiring
(source-inspection style, matching the RD04/RD12/RD13 test files'
established pattern). Design reviewed by GPT (req_3bf3e04fec03449a)
before implementation."""

from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.patch import validation_campaign as vc  # noqa: E402


class Rd26ContractCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.run_source = inspect.getsource(vc.run)
        self.main_source = inspect.getsource(vc.main)

    def test_flag_exists_and_defaults_false(self) -> None:
        self.assertIn('"--run-rd26-contract"', self.main_source)

    def _block(self) -> str:
        block_start = self.run_source.index("if args.run_rd26_contract:")
        return self.run_source[block_start:self.run_source.index(
            "validation_check_results: dict[str, object] = {}", block_start
        )]

    def test_rd26_only_gating(self) -> None:
        self.assertIn(
            'descriptor.experiment_contract != "RD26-DECODE-VERIFY-BIT-IDENTITY"',
            self._block(),
        )

    def test_calls_run_rd26_decode_verify_bit_identity_check(self) -> None:
        self.assertIn(
            "run_rd26_decode_verify_bit_identity_check(", self._block(),
        )

    def test_uses_bit_identical_result(self) -> None:
        self.assertIn(
            'rd26_qualification["results"]["bit_identical"]', self._block(),
        )

    def test_binds_canonical_correctness_json_never_raw_artifact(self) -> None:
        block = self._block()
        self.assertNotIn(
            'correctness_evidence = {"artifact": rd26_qualification["artifact"]}',
            block,
        )
        self.assertIn(
            '"path": correctness_path.relative_to(campaign_run_dir).as_posix()',
            block,
        )
        self.assertIn(
            '"sha256": hashlib.sha256(correctness_path.read_bytes()).hexdigest()',
            block,
        )

    def test_never_writes_contract_promotions(self) -> None:
        self.assertNotIn("contract_promotions[", self._block())

    def test_does_not_synthesize_activation_evidence(self) -> None:
        # RD26's contract requires correctness+controls only (expected_effect
        # = "correctness", no performance claim) -- no activation capability
        # is required, so this block must never fabricate ActivationEvidence.
        self.assertNotIn("activation_evidence = ActivationEvidence(", self._block())

    def test_run_rd26_check_exposes_nested_validation_build_identities(self) -> None:
        source = inspect.getsource(vc.run_rd26_decode_verify_bit_identity_check)
        self.assertIn('"validation_build_identities": {', source)
        self.assertIn('"control": artifact_doc["control_build_identity"]', source)
        self.assertIn('"subject": artifact_doc["subject_build_identity"]', source)

    def test_validation_build_identities_thread_rd26_qualification(self) -> None:
        self.assertIn(
            'else rd26_qualification["validation_build_identities"]\n'
            "            if rd26_qualification is not None",
            self.run_source,
        )

    def test_contract_correctness_gate_threads_rd26_named_results(self) -> None:
        self.assertIn(
            "rd26_correctness_named_results = (\n"
            "            rd26_qualification[\"results\"] if rd26_qualification is not None else None\n"
            "        )",
            self.run_source,
        )
        gate_call_start = self.run_source.index(
            "contract_correctness_gate = compute_contract_correctness_gate("
        )
        gate_call = self.run_source[gate_call_start:gate_call_start + 1200]
        self.assertIn("else rd26_correctness_named_results", gate_call)

    def test_present_in_standard_campaign_skip(self) -> None:
        self.assertIn(
            "args.run_rd12_contract or args.run_rd04_contract or "
            "args.run_rd13_contract or args.run_rd26_contract):",
            self.run_source,
        )

    def test_present_in_run_performance_benchmark_exclusion(self) -> None:
        self.assertIn(
            '"run_rd04_contract", "run_rd13_contract", "run_rd26_contract",',
            self.main_source,
        )

    def test_present_in_framework_configuration_exclusion(self) -> None:
        source = inspect.getsource(vc._run_framework_configuration)
        self.assertIn(
            '"run_rd58_state_restore", "run_rd73_contract", "run_rd12_contract", '
            '"run_rd04_contract", "run_rd13_contract", "run_rd26_contract",',
            source,
        )


if __name__ == "__main__":
    unittest.main()
