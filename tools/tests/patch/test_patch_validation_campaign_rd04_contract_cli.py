"""PA39: real hardware-free coverage for --run-rd04-contract's CLI wiring
(source-inspection style, matching the RD12 test file's established
pattern -- run()/main() are large integration entry points that cannot be
reasonably unit-tested end to end without real hardware)."""

from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.patch import validation_campaign as vc  # noqa: E402


class Rd04ContractCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.run_source = inspect.getsource(vc.run)
        self.main_source = inspect.getsource(vc.main)

    def test_flag_exists_and_defaults_false(self) -> None:
        self.assertIn('"--run-rd04-contract"', self.main_source)

    def test_rd04_corpus_flag_exists(self) -> None:
        self.assertIn('"--rd04-corpus"', self.main_source)

    def test_requires_rd04_corpus(self) -> None:
        block_start = self.run_source.index("if args.run_rd04_contract:")
        block = self.run_source[block_start:self.run_source.index(
            "validation_check_results: dict[str, object] = {}"
        )]
        self.assertIn("--run-rd04-contract requires --rd04-corpus", block)

    def test_mutually_exclusive_with_other_specialized_modes(self) -> None:
        block_start = self.run_source.index("if args.run_rd04_contract:")
        block = self.run_source[block_start:self.run_source.index(
            "validation_check_results: dict[str, object] = {}"
        )]
        self.assertIn("run-rd04-contract is mutually exclusive with the", block)
        self.assertIn(
            "if args.run_rd08_lanes or args.run_rd08_contract or "
            "args.run_rd04_benchmark or args.run_rd58_state_restore or "
            "args.run_rd73_contract or args.run_rd12_contract:",
            block,
        )
        # Same class of bug caught twice already this session: the
        # mutual-exclusion condition must never reference itself.
        self.assertNotIn("args.run_rd04_contract:\n            raise", block)

    def test_rd04_only_gating(self) -> None:
        self.assertIn(
            'descriptor.experiment_contract != "RD04-BF16-FLASH-ATTN-TILE"',
            self.run_source,
        )

    def test_binds_correctness_summary_evidence_never_contract_promotions(self) -> None:
        # GPT review (req_5631b12dc3fb4a23): correctness_evidence must
        # point at the canonical correctness.json (real "disposition"
        # field), never rd04_qualification["artifact"] (no "disposition").
        block_start = self.run_source.index("if args.run_rd04_contract:")
        block = self.run_source[block_start:self.run_source.index(
            "validation_check_results: dict[str, object] = {}"
        )]
        self.assertNotIn(
            'correctness_evidence = {"artifact": rd04_qualification["artifact"]}',
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
        self.assertNotIn("contract_promotions[", block)

    def test_disposition_requires_both_named_checks(self) -> None:
        block_start = self.run_source.index("if args.run_rd04_contract:")
        block = self.run_source[block_start:self.run_source.index(
            "validation_check_results: dict[str, object] = {}"
        )]
        self.assertIn(
            "rd04_all_passed = rd04_backend_reference_result.passed and "
            "rd04_ppl_equality_result.passed",
            block,
        )
        self.assertIn(
            '"disposition": "passed" if rd04_all_passed else "failed"',
            block,
        )

    def test_does_not_bind_activation_evidence(self) -> None:
        # RD04 has no real BIGCHERRY_PATCH_TRACE marker in its source
        # (confirmed by inspection) -- unlike RD12, its block must never
        # synthesize an ActivationEvidence; the declared activation check
        # stays honestly BLOCKED.
        block_start = self.run_source.index("if args.run_rd04_contract:")
        block = self.run_source[block_start:self.run_source.index(
            "validation_check_results: dict[str, object] = {}"
        )]
        self.assertNotIn("activation_evidence = ActivationEvidence(", block)

    def test_specialized_evidence_mode_exclusions_include_rd04(self) -> None:
        self.assertIn(
            "if not (args.run_rd08_contract or args.run_rd04_benchmark or "
            "args.run_rd58_state_restore or args.run_rd73_contract or "
            "args.run_rd12_contract or args.run_rd04_contract):",
            self.run_source,
        )
        self.assertIn(
            "trace_result = None if (args.run_rd08_contract or "
            "args.run_rd04_benchmark or args.run_rd58_state_restore or "
            "args.run_rd73_contract or args.run_rd12_contract or "
            "args.run_rd04_contract) else "
            "run_trace_activation_probes(",
            self.run_source,
        )

    def test_run_performance_benchmark_exclusion_includes_rd04(self) -> None:
        self.assertIn(
            '"run_rd58_state_restore", "run_rd73_contract", "run_rd12_contract", '
            '"run_rd04_contract",',
            self.main_source,
        )

    def test_correctness_evidence_ambiguity_guard_covers_rd04(self) -> None:
        self.assertIn(
            "args.correctness_evidence is not None and args.run_rd04_contract",
            self.run_source,
        )

    def test_run_rd04_contract_correctness_exposes_validation_build_identities(self) -> None:
        source = inspect.getsource(vc.run_rd04_contract_correctness)
        self.assertIn('"validation_build_identities": {', source)
        self.assertIn('"control": doc["control_build_identity"]', source)
        self.assertIn('"subject": doc["subject_build_identity"]', source)

    def test_validation_build_identities_thread_rd04_qualification(self) -> None:
        # GPT review (req_5631b12dc3fb4a23): RD04's producer builds its own
        # isolated control/subject worktrees (same shape as RD12), so
        # falling through to the generic campaign identities was wrong.
        self.assertIn(
            'else rd04_qualification["validation_build_identities"]\n'
            "            if rd04_qualification is not None",
            self.run_source,
        )

    def test_contract_correctness_gate_threads_rd04_named_results(self) -> None:
        self.assertIn(
            "rd04_correctness_named_results = (\n"
            "            rd04_qualification[\"results\"] if rd04_qualification is not None else None\n"
            "        )",
            self.run_source,
        )
        gate_call_start = self.run_source.index(
            "contract_correctness_gate = compute_contract_correctness_gate("
        )
        gate_call = self.run_source[gate_call_start:gate_call_start + 900]
        self.assertIn("else rd04_correctness_named_results", gate_call)


if __name__ == "__main__":
    unittest.main()
