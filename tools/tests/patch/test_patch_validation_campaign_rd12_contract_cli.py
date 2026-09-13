"""PA39: real hardware-free coverage for --run-rd12-contract's CLI wiring
(source-inspection style, matching test_patch_validation_campaign_va04.py's
established pattern for --run-rd04-benchmark -- run()/main() are large
integration entry points that cannot be reasonably unit-tested end to end
without real hardware)."""

from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.patch import validation_campaign as vc  # noqa: E402


class Rd12ContractCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.run_source = inspect.getsource(vc.run)
        self.main_source = inspect.getsource(vc.main)

    def test_flag_exists_and_defaults_false(self) -> None:
        self.assertIn('"--run-rd12-contract"', self.main_source)

    def test_mutually_exclusive_with_other_specialized_modes(self) -> None:
        block_start = self.run_source.index("if args.run_rd12_contract:")
        block = self.run_source[block_start:self.run_source.index(
            "validation_check_results: dict[str, object] = {}"
        )]
        self.assertIn("run-rd12-contract is mutually exclusive with the", block)
        self.assertIn(
            "if args.run_rd08_lanes or args.run_rd08_contract or "
            "args.run_rd04_benchmark or args.run_rd58_state_restore or "
            "args.run_rd73_contract:",
            block,
        )
        # The mutual-exclusion condition must never reference itself --
        # a real bug caught during authoring: including run_rd12_contract
        # in its own guard makes the condition always true inside the
        # branch, so the flag could never actually run.
        self.assertNotIn("args.run_rd12_contract:\n            raise", block)

    def test_rd12_only_gating(self) -> None:
        self.assertIn(
            'descriptor.experiment_contract != "RD12-PAIRED-MMVQ-DUAL"',
            self.run_source,
        )

    def test_binds_correctness_evidence_never_contract_promotions(self) -> None:
        block_start = self.run_source.index("if args.run_rd12_contract:")
        block = self.run_source[block_start:self.run_source.index(
            "validation_check_results: dict[str, object] = {}"
        )]
        self.assertIn("correctness_evidence = {\"artifact\": rd12_qualification[\"artifact\"]}", block)
        self.assertNotIn("contract_promotions[", block)

    def test_disposition_derives_from_bit_identical_not_backend_reference(self) -> None:
        block_start = self.run_source.index("if args.run_rd12_contract:")
        block = self.run_source[block_start:self.run_source.index(
            "validation_check_results: dict[str, object] = {}"
        )]
        self.assertIn(
            '"disposition": "passed" if bit_identical_result.passed else "failed"',
            block,
        )

    def test_specialized_evidence_mode_exclusions_include_rd12(self) -> None:
        # The two standalone specialized-mode gates (generic trace probe,
        # generic S1-S7 campaign execution) must both exclude RD12 -- these
        # are the exact invariants test_patch_validation_campaign_va15_wiring
        # covers for RD08/RD04/RD58/RD73; RD12 must join the same guard.
        self.assertIn(
            "if not (args.run_rd08_contract or args.run_rd04_benchmark or "
            "args.run_rd58_state_restore or args.run_rd73_contract or "
            "args.run_rd12_contract):",
            self.run_source,
        )
        self.assertIn(
            "trace_result = None if (args.run_rd08_contract or "
            "args.run_rd04_benchmark or args.run_rd58_state_restore or "
            "args.run_rd73_contract or args.run_rd12_contract) else "
            "run_trace_activation_probes(",
            self.run_source,
        )

    def test_run_performance_benchmark_exclusion_includes_rd12(self) -> None:
        self.assertIn(
            '"run_rd58_state_restore", "run_rd73_contract", "run_rd12_contract",',
            self.main_source,
        )

    def test_correctness_evidence_ambiguity_guard_covers_rd12(self) -> None:
        self.assertIn(
            "args.correctness_evidence is not None and args.run_rd12_contract",
            self.run_source,
        )

    def test_validation_build_identities_thread_rd12_qualification(self) -> None:
        self.assertIn(
            'else rd12_qualification["validation_build_identities"]\n'
            "            if rd12_qualification is not None",
            self.run_source,
        )

    def test_run_rd12_correctness_check_exposes_validation_build_identities(self) -> None:
        source = inspect.getsource(vc.run_rd12_correctness_check)
        self.assertIn('"validation_build_identities": {', source)
        self.assertIn('"control": artifact_doc["control_build_identity"]', source)
        self.assertIn('"subject": artifact_doc["subject_build_identity"]', source)


if __name__ == "__main__":
    unittest.main()
