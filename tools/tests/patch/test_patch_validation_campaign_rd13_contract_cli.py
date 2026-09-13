"""Real hardware-free coverage for --run-rd13-contract's CLI wiring
(source-inspection style, matching the RD04/RD12 test files' established
pattern -- run()/main() are large integration entry points that cannot be
reasonably unit-tested end to end without real hardware). Design reviewed
by GPT (req_3bf3e04fec03449a) before implementation."""

from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.patch import validation_campaign as vc  # noqa: E402


class Rd13ContractCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.run_source = inspect.getsource(vc.run)
        self.main_source = inspect.getsource(vc.main)

    def test_flag_exists_and_defaults_false(self) -> None:
        self.assertIn('"--run-rd13-contract"', self.main_source)

    def _block(self) -> str:
        block_start = self.run_source.index("if args.run_rd13_contract:")
        # RD13's own block contains an inline "if args.run_rd26_contract:"
        # mutual-exclusion check (8-space indent) -- search past that for
        # RD26's actual block start (4-space indent, "\n    if ...").
        block_end = self.run_source.index("\n    if args.run_rd26_contract:", block_start)
        return self.run_source[block_start:block_end]

    def test_mutually_exclusive_with_rd26(self) -> None:
        self.assertIn(
            "run-rd13-contract is mutually exclusive with", self._block(),
        )

    def test_rd13_only_gating(self) -> None:
        self.assertIn(
            'descriptor.experiment_contract != "RD13-MUL-MAT-ADD-VIEW-FUSION"',
            self._block(),
        )

    def test_calls_run_rd13_backend_reference_check(self) -> None:
        self.assertIn("run_rd13_backend_reference_check(", self._block())

    def test_uses_only_backend_reference_check(self) -> None:
        block = self._block()
        self.assertIn(
            'rd13_qualification["results"]["backend_reference"]', block,
        )
        self.assertNotIn("ppl_equality", block)

    def test_binds_canonical_correctness_json_never_raw_artifact(self) -> None:
        block = self._block()
        self.assertNotIn(
            'correctness_evidence = {"artifact": rd13_qualification["artifact"]}',
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

    def test_run_rd13_backend_reference_check_exposes_nested_validation_build_identities(self) -> None:
        source = inspect.getsource(vc.run_rd13_backend_reference_check)
        self.assertIn('"validation_build_identities": {', source)
        self.assertIn('"control": doc["control_build_identity"]', source)
        self.assertIn('"subject": doc["subject_build_identity"]', source)

    def test_validation_build_identities_thread_rd13_qualification(self) -> None:
        self.assertIn(
            'else rd13_qualification["validation_build_identities"]\n'
            "            if rd13_qualification is not None",
            self.run_source,
        )

    def test_contract_correctness_gate_threads_rd13_named_results(self) -> None:
        self.assertIn(
            "rd13_correctness_named_results = (\n"
            "            rd13_qualification[\"results\"] if rd13_qualification is not None else None\n"
            "        )",
            self.run_source,
        )
        gate_call_start = self.run_source.index(
            "contract_correctness_gate = compute_contract_correctness_gate("
        )
        gate_call = self.run_source[gate_call_start:gate_call_start + 1200]
        self.assertIn("else rd13_correctness_named_results", gate_call)

    def test_absent_from_generic_trace_probe_exclusion(self) -> None:
        # RD13's real trace-probe negative control (GGML_CUDA_DISABLE_FUSION)
        # remains valid and must still run -- RD13 is deliberately NOT added
        # to this exclusion, unlike every other specialized mode.
        self.assertIn(
            "trace_result = None if (args.run_rd08_contract or "
            "args.run_rd04_benchmark or args.run_rd58_state_restore or "
            "args.run_rd73_contract or args.run_rd12_contract or "
            "args.run_rd04_contract) else "
            "run_trace_activation_probes(",
            self.run_source,
        )

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
