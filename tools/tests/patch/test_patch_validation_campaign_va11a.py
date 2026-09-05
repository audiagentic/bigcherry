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


# RV95: compute_persisted_validation_eligible() now also requires the record's
# own activation/correctness dispositions, matching evidence.py's
# _record_qualifies(). These are the passing values; the tests below isolate
# the adapter/contract axis, so they hold this axis good on purpose.
_ACT_OK = "activation-verified"
_CORR_OK = {"disposition": "passed"}


class ComputePersistedValidationEligibleTests(unittest.TestCase):
    """VA14 final slice: a bound-contract patch is eligible only when BOTH
    the adapter verdict AND every bound contract's own promotion result
    passed -- not the adapter verdict alone (GPT round 8,
    req_84fca34f83064678) and not merely "a promotion happened" without
    checking its `passed` field."""

    def test_bound_contract_with_no_promotion_at_all_is_false(self) -> None:
        descriptor = _FakeDescriptor(("RD08-Q6K-MMVQ-VDR2",))
        result = vc.compute_persisted_validation_eligible(
            descriptor, _FakeVerdict(eligible=True), None,
            activation_disposition=_ACT_OK, correctness=_CORR_OK,
        )
        self.assertFalse(result)

    def test_bound_contract_with_no_verdict_at_all_is_false(self) -> None:
        descriptor = _FakeDescriptor(("RD08-Q6K-MMVQ-VDR2",))
        self.assertFalse(vc.compute_persisted_validation_eligible(
            descriptor, None, {}, activation_disposition=_ACT_OK, correctness=_CORR_OK,
        ))

    def test_bound_contract_adapter_pass_and_promotion_pass_is_true(self) -> None:
        descriptor = _FakeDescriptor(("RD08-Q6K-MMVQ-VDR2",))
        promotions = {"RD08-Q6K-MMVQ-VDR2": {"passed": True}}
        result = vc.compute_persisted_validation_eligible(
            descriptor, _FakeVerdict(eligible=True), promotions,
            activation_disposition=_ACT_OK, correctness=_CORR_OK,
        )
        self.assertTrue(result)

    def test_bound_contract_adapter_pass_but_promotion_fail_is_false(self) -> None:
        descriptor = _FakeDescriptor(("RD08-Q6K-MMVQ-VDR2",))
        promotions = {"RD08-Q6K-MMVQ-VDR2": {"passed": False}}
        result = vc.compute_persisted_validation_eligible(
            descriptor, _FakeVerdict(eligible=True), promotions,
            activation_disposition=_ACT_OK, correctness=_CORR_OK,
        )
        self.assertFalse(result)

    def test_bound_contract_promotion_pass_but_adapter_fail_is_false(self) -> None:
        descriptor = _FakeDescriptor(("RD08-Q6K-MMVQ-VDR2",))
        promotions = {"RD08-Q6K-MMVQ-VDR2": {"passed": True}}
        result = vc.compute_persisted_validation_eligible(
            descriptor, _FakeVerdict(eligible=False), promotions,
            activation_disposition=_ACT_OK, correctness=_CORR_OK,
        )
        self.assertFalse(result)

    def test_multi_contract_incomplete_promotion_set_is_false(self) -> None:
        descriptor = _FakeDescriptor(("RD08-Q6K-MMVQ-VDR2", "RD04-BF16-FLASH-ATTN-TILE"))
        promotions = {"RD08-Q6K-MMVQ-VDR2": {"passed": True}}
        result = vc.compute_persisted_validation_eligible(
            descriptor, _FakeVerdict(eligible=True), promotions,
            activation_disposition=_ACT_OK, correctness=_CORR_OK,
        )
        self.assertFalse(result)

    def test_multi_contract_complete_promotion_set_is_true(self) -> None:
        descriptor = _FakeDescriptor(("RD08-Q6K-MMVQ-VDR2", "RD04-BF16-FLASH-ATTN-TILE"))
        promotions = {
            "RD08-Q6K-MMVQ-VDR2": {"passed": True},
            "RD04-BF16-FLASH-ATTN-TILE": {"passed": True},
        }
        result = vc.compute_persisted_validation_eligible(
            descriptor, _FakeVerdict(eligible=True), promotions,
            activation_disposition=_ACT_OK, correctness=_CORR_OK,
        )
        self.assertTrue(result)

    def test_no_contract_passes_through_the_real_adapter_verdict_unaffected(self) -> None:
        descriptor = _FakeDescriptor(())
        self.assertTrue(
            vc.compute_persisted_validation_eligible(
                descriptor, _FakeVerdict(eligible=True), {},
                activation_disposition=_ACT_OK, correctness=_CORR_OK,
            )
        )
        self.assertFalse(
            vc.compute_persisted_validation_eligible(
                descriptor, _FakeVerdict(eligible=False), {},
                activation_disposition=_ACT_OK, correctness=_CORR_OK,
            )
        )

    def test_no_contract_no_verdict_returns_none(self) -> None:
        descriptor = _FakeDescriptor(())
        self.assertIsNone(vc.compute_persisted_validation_eligible(
            descriptor, None, {}, activation_disposition=_ACT_OK, correctness=_CORR_OK,
        ))


class EligibilityAgreesWithTheEvidenceVerifierTests(unittest.TestCase):
    """RV95: this flag and evidence.py's verify_validated_patch() answer the
    same question and must not be able to disagree. The regression they pin
    is REAL and was observed on hardware: --run-rd73-contract produced a
    record with a passing adapter verdict and a passing contract promotion,
    but left the record's own activation/correctness at
    disposition="unknown". The campaign printed "STATE='validated' eligible:
    yes" for a record verify_validated_patch() rejected with "activation is
    not executed+activation-verified; correctness did not pass".

    Fail-OPEN is the specific direction that matters here: an over-strict
    flag merely blocks a promotion a human can re-run, whereas an
    over-permissive one advertises a qualification the verifier will not
    honour."""

    def _eligible(self, **overrides):
        kwargs = {
            "activation_disposition": _ACT_OK,
            "correctness": _CORR_OK,
        }
        kwargs.update(overrides)
        return vc.compute_persisted_validation_eligible(
            _FakeDescriptor(("RD73-STABLE-GRAPH-CACHE-KEY",)),
            _FakeVerdict(eligible=True),
            {"RD73-STABLE-GRAPH-CACHE-KEY": {"passed": True}},
            **kwargs,
        )

    def test_unknown_activation_disposition_is_not_eligible(self) -> None:
        # The exact RD73 record shape: everything else passes.
        self.assertFalse(self._eligible(activation_disposition="unknown"))

    def test_unknown_correctness_disposition_is_not_eligible(self) -> None:
        self.assertFalse(self._eligible(correctness={"disposition": "unknown"}))

    def test_missing_correctness_entirely_is_not_eligible(self) -> None:
        # correctness_summary stays None when a producer never writes it.
        self.assertFalse(self._eligible(correctness=None))

    def test_failed_correctness_is_not_eligible(self) -> None:
        self.assertFalse(self._eligible(correctness={"disposition": "failed"}))

    def test_failed_activation_is_not_eligible(self) -> None:
        self.assertFalse(self._eligible(activation_disposition="failed-activation"))

    def test_all_axes_passing_is_eligible(self) -> None:
        self.assertTrue(self._eligible())

    def test_required_literals_match_the_evidence_verifier_exactly(self) -> None:
        # If evidence.py ever renames these dispositions, the two predicates
        # would silently diverge again -- which is the whole defect. Assert
        # the accepted values against the verifier's own source of truth.
        import inspect

        from bigcherry.patch import evidence

        source = inspect.getsource(evidence._record_qualifies)
        self.assertIn('!= "activation-verified"', source)
        self.assertIn('!= "passed"', source)


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
