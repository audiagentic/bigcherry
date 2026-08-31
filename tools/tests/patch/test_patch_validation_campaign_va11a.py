"""VA11A/VA14 tests: compute_contract_correctness_gate() and
compute_persisted_validation_eligible() -- the real, named-correctness
independence check and the final contract-promotion-gated eligibility
composition. Real contract ids loaded from the actual
config/experiment-contracts.toml, not synthetic fixtures, so this proves
the fix against the real data the campaign will actually see.
"""

from __future__ import annotations

import sys
import unittest
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.experiment import contract as ec  # noqa: E402
from bigcherry.patch import validation_campaign as vc  # noqa: E402

_CONTRACTS = ec.load_contracts(
    Path(__file__).resolve().parents[3] / "config" / "experiment-contracts.toml"
)
RD08 = _CONTRACTS.contracts["RD08-Q6K-MMVQ-VDR2"]
RD04 = _CONTRACTS.contracts["RD04-BF16-FLASH-ATTN-TILE"]


class ComputeContractCorrectnessGateTests(unittest.TestCase):
    def test_no_bound_contract_returns_none(self) -> None:
        self.assertIsNone(vc.compute_contract_correctness_gate(None, None))

    def test_single_required_check_no_results_is_blocked(self) -> None:
        # RD08-Q6K-MMVQ-VDR2 requires exactly one check (bit_identical).
        gate = vc.compute_contract_correctness_gate(RD08, None)
        self.assertIsNotNone(gate)
        self.assertFalse(gate["passed"])
        self.assertIn("bit_identical", gate["missing_checks"])

    def test_single_required_check_passes_with_a_real_named_result(self) -> None:
        result = ec.CorrectnessResult(check="bit_identical", passed=True, detail="15/15 rows ok")
        gate = vc.compute_contract_correctness_gate(RD08, {"bit_identical": result})
        self.assertIsNotNone(gate)
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["missing_checks"], [])
        self.assertEqual(gate["failed_checks"], [])

    def test_single_required_check_fails_with_a_failing_named_result(self) -> None:
        result = ec.CorrectnessResult(check="bit_identical", passed=False, detail="mismatch")
        gate = vc.compute_contract_correctness_gate(RD08, {"bit_identical": result})
        self.assertIsNotNone(gate)
        self.assertFalse(gate["passed"])
        self.assertIn("bit_identical", gate["failed_checks"])

    def test_generic_summary_cannot_satisfy_a_named_check(self) -> None:
        # A dict that isn't a real CorrectnessResult (e.g. a stray generic
        # --correctness-evidence summary shape) must never be silently
        # accepted as satisfying a named check -- only real CorrectnessResult
        # objects with .passed are read.
        with self.assertRaises(AttributeError):
            vc.compute_contract_correctness_gate(
                RD08, {"bit_identical": {"disposition": "passed"}}
            )

    def test_two_required_checks_blocked_never_fabricates_two_passes(self) -> None:
        # RD04's contract requires BOTH backend_reference AND ppl_equality
        # independently -- one named result must never silently cover both.
        result = ec.CorrectnessResult(check="backend_reference", passed=True)
        gate = vc.compute_contract_correctness_gate(RD04, {"backend_reference": result})
        self.assertIsNotNone(gate)
        self.assertFalse(gate["passed"])
        self.assertIn("ppl_equality", gate["missing_checks"])
        self.assertNotIn("backend_reference", gate["missing_checks"])

    def test_two_required_checks_both_pass(self) -> None:
        results = {
            "backend_reference": ec.CorrectnessResult(check="backend_reference", passed=True),
            "ppl_equality": ec.CorrectnessResult(check="ppl_equality", passed=True),
        }
        gate = vc.compute_contract_correctness_gate(RD04, results)
        self.assertTrue(gate["passed"])


@dataclass
class _FakeDescriptor:
    experiment_contracts: tuple[str, ...]

    @property
    def experiment_contract(self) -> str | None:
        if not self.experiment_contracts:
            return None
        return self.experiment_contracts[0]


@dataclass
class _FakeVerdict:
    eligible: bool


class ComputePersistedValidationEligibleTests(unittest.TestCase):
    """VA14 final slice: a bound-contract patch is eligible only when BOTH
    the adapter verdict AND every bound contract's own promotion result
    passed -- not the adapter verdict alone (GPT round 8,
    req_84fca34f83064678) and not merely "a promotion happened" without
    checking its `passed` field."""

    def test_bound_contract_with_no_promotion_at_all_is_false(self) -> None:
        descriptor = _FakeDescriptor(("RD08-Q6K-MMVQ-VDR2",))
        result = vc.compute_persisted_validation_eligible(
            descriptor, _FakeVerdict(eligible=True), None
        )
        self.assertFalse(result)

    def test_bound_contract_with_no_verdict_at_all_is_false(self) -> None:
        descriptor = _FakeDescriptor(("RD08-Q6K-MMVQ-VDR2",))
        self.assertFalse(vc.compute_persisted_validation_eligible(descriptor, None, {}))

    def test_bound_contract_adapter_pass_and_promotion_pass_is_true(self) -> None:
        descriptor = _FakeDescriptor(("RD08-Q6K-MMVQ-VDR2",))
        promotions = {"RD08-Q6K-MMVQ-VDR2": {"passed": True}}
        result = vc.compute_persisted_validation_eligible(
            descriptor, _FakeVerdict(eligible=True), promotions
        )
        self.assertTrue(result)

    def test_bound_contract_adapter_pass_but_promotion_fail_is_false(self) -> None:
        descriptor = _FakeDescriptor(("RD08-Q6K-MMVQ-VDR2",))
        promotions = {"RD08-Q6K-MMVQ-VDR2": {"passed": False}}
        result = vc.compute_persisted_validation_eligible(
            descriptor, _FakeVerdict(eligible=True), promotions
        )
        self.assertFalse(result)

    def test_bound_contract_promotion_pass_but_adapter_fail_is_false(self) -> None:
        descriptor = _FakeDescriptor(("RD08-Q6K-MMVQ-VDR2",))
        promotions = {"RD08-Q6K-MMVQ-VDR2": {"passed": True}}
        result = vc.compute_persisted_validation_eligible(
            descriptor, _FakeVerdict(eligible=False), promotions
        )
        self.assertFalse(result)

    def test_multi_contract_incomplete_promotion_set_is_false(self) -> None:
        descriptor = _FakeDescriptor(("RD08-Q6K-MMVQ-VDR2", "RD04-BF16-FLASH-ATTN-TILE"))
        promotions = {"RD08-Q6K-MMVQ-VDR2": {"passed": True}}
        result = vc.compute_persisted_validation_eligible(
            descriptor, _FakeVerdict(eligible=True), promotions
        )
        self.assertFalse(result)

    def test_multi_contract_complete_promotion_set_is_true(self) -> None:
        descriptor = _FakeDescriptor(("RD08-Q6K-MMVQ-VDR2", "RD04-BF16-FLASH-ATTN-TILE"))
        promotions = {
            "RD08-Q6K-MMVQ-VDR2": {"passed": True},
            "RD04-BF16-FLASH-ATTN-TILE": {"passed": True},
        }
        result = vc.compute_persisted_validation_eligible(
            descriptor, _FakeVerdict(eligible=True), promotions
        )
        self.assertTrue(result)

    def test_no_contract_passes_through_the_real_adapter_verdict_unaffected(self) -> None:
        descriptor = _FakeDescriptor(())
        self.assertTrue(
            vc.compute_persisted_validation_eligible(descriptor, _FakeVerdict(eligible=True), {})
        )
        self.assertFalse(
            vc.compute_persisted_validation_eligible(descriptor, _FakeVerdict(eligible=False), {})
        )

    def test_no_contract_no_verdict_returns_none(self) -> None:
        descriptor = _FakeDescriptor(())
        self.assertIsNone(vc.compute_persisted_validation_eligible(descriptor, None, {}))


class BoundArtifactShapesActuallyPassTests(unittest.TestCase):
    """Proves the trace_evidence/correctness_evidence shapes VA11A now
    builds in run() are not just plausible-looking dicts -- they make the
    real built-in validators (_builtin_trace_marker, _builtin_backend_ops)
    actually reach PASS, using real files on disk exactly as run() would
    produce them. Before this fix, trace_evidence={} always BLOCKED
    trace-marker checks, and correctness_evidence being the raw decoded
    dict (no "artifact" key) always BLOCKED backend-ops checks."""

    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self.run_dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _bind(self, relative_path: str) -> dict:
        import hashlib
        target = self.run_dir / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        return {"path": relative_path, "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}

    def test_trace_marker_check_passes_with_the_real_shape(self) -> None:
        from bigcherry.patch import validation as pv

        pos_log = self.run_dir / "logs" / "activation-positive.log"
        pos_log.parent.mkdir(parents=True, exist_ok=True)
        pos_log.write_text("BIGCHERRY_PATCH_HIT patch=x\n", encoding="utf-8")
        neg_log = self.run_dir / "logs" / "activation-fusion-disabled.log"
        neg_log.write_text("nothing here\n", encoding="utf-8")

        trace_evidence = {
            "positive": {"marker_regex": "BIGCHERRY_PATCH_HIT", "artifact": self._bind("logs/activation-positive.log")},
            "negative": {"marker_regex": "BIGCHERRY_PATCH_HIT", "artifact": self._bind("logs/activation-fusion-disabled.log")},
        }
        ctx = pv.ValidationContext(
            descriptor=None, base_revision="a" * 40, control_source=None, subject_source=None,
            run_dir=self.run_dir, trace_evidence=trace_evidence,
        )
        spec = pv.CheckSpec("activation", "activation", "trace-marker", True,
                             {"marker-regex": "BIGCHERRY_PATCH_HIT"})
        result = pv.evaluate_check(spec, ctx)
        self.assertEqual(result.status, pv.PASS, result.summary)

    def test_backend_ops_check_reaches_a_real_verdict_with_the_real_shape(self) -> None:
        from bigcherry.patch import validation as pv
        import json

        correctness_path = self.run_dir / "correctness.json"
        correctness_path.write_text(
            json.dumps({"ops": ["MUL_MAT"], "passed": True}), encoding="utf-8"
        )
        correctness_evidence = {"artifact": self._bind("correctness.json")}
        ctx = pv.ValidationContext(
            descriptor=None, base_revision="a" * 40, control_source=None, subject_source=None,
            run_dir=self.run_dir, correctness_evidence=correctness_evidence,
        )
        spec = pv.CheckSpec("correctness", "correctness", "backend-ops", True, {"ops": ["MUL_MAT"]})
        result = pv.evaluate_check(spec, ctx)
        # Before this fix (correctness_evidence as a raw dict with no
        # "artifact" key) this was UNCONDITIONALLY BLOCKED regardless of
        # content -- the real assertion is that it now reaches a real
        # PASS/FAIL verdict instead of always BLOCKED.
        self.assertNotEqual(result.status, pv.BLOCKED, result.summary)
        self.assertEqual(result.status, pv.PASS, result.summary)


if __name__ == "__main__":
    unittest.main()
