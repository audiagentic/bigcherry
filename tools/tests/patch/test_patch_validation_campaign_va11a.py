"""VA11A tests: compute_contract_correctness_gate() -- the real, RD04-style
named-correctness independence check. Real contract ids loaded from the
actual config/experiment-contracts.toml, not synthetic fixtures, so this
proves the fix against the real data the campaign will actually see.
"""

from __future__ import annotations

import sys
import unittest
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.patch import validation_campaign as vc # noqa: E402


@dataclass
class _FakeDescriptor:
    experiment_contract: str | None


class ComputeContractCorrectnessGateTests(unittest.TestCase):
    def test_no_bound_contract_returns_none(self) -> None:
        self.assertIsNone(vc.compute_contract_correctness_gate(_FakeDescriptor(None), None))

    def test_single_required_check_no_summary_is_blocked(self) -> None:
        # RD08-Q6K-MMVQ-VDR2 requires exactly one check (bit_identical).
        descriptor = _FakeDescriptor("RD08-Q6K-MMVQ-VDR2")
        gate = vc.compute_contract_correctness_gate(descriptor, None)
        self.assertIsNotNone(gate)
        self.assertEqual(gate.get("status"), "blocked")
        self.assertFalse(gate["passed"])

    def test_single_required_check_is_ALWAYS_blocked_even_with_a_passing_summary(self) -> None:
        """GPT round 8 (req_84fca34f83064678): the earlier version of this
        function treated a single-required-check contract as provable from
        the one generic --correctness-evidence summary. That summary has no
        machine-readable field naming WHICH check it proves -- an arbitrary
        passing summary could silently become 'bit_identical passed' without
        ever proving bit-identical was what actually ran. Fixed: ANY
        non-empty required_checks is blocked, one name or several, until a
        real per-named-check evidence producer exists (VA14)."""
        descriptor = _FakeDescriptor("RD08-Q6K-MMVQ-VDR2")
        gate = vc.compute_contract_correctness_gate(
            descriptor, {"disposition": "passed", "detail": "bit-identical vs native"}
        )
        self.assertIsNotNone(gate)
        self.assertEqual(gate.get("status"), "blocked")
        self.assertFalse(gate["passed"])
        self.assertIn("bit_identical", gate["detail"])

    def test_two_required_checks_is_blocked_never_fabricates_two_passes(self) -> None:
        """The real bug this item fixes: RD04's contract requires BOTH
        backend_reference AND ppl_equality independently. There is no
        per-named-check evidence input mechanism yet -- one generic
        correctness_summary must never be read as satisfying both."""
        descriptor = _FakeDescriptor("RD04-BF16-FLASH-ATTN-TILE")
        gate = vc.compute_contract_correctness_gate(
            descriptor, {"disposition": "passed", "detail": "looks fine"}
        )
        self.assertIsNotNone(gate)
        self.assertFalse(gate["passed"])
        self.assertEqual(gate.get("status"), "blocked")
        self.assertIn("backend_reference", gate["detail"])
        self.assertIn("ppl_equality", gate["detail"])

    def test_two_required_checks_blocked_even_with_no_summary(self) -> None:
        descriptor = _FakeDescriptor("RD04-BF16-FLASH-ATTN-TILE")
        gate = vc.compute_contract_correctness_gate(descriptor, None)
        self.assertIsNotNone(gate)
        self.assertEqual(gate.get("status"), "blocked")

    def test_unknown_contract_id_raises(self) -> None:
        descriptor = _FakeDescriptor("NOT-A-REAL-CONTRACT-ID")
        with self.assertRaises(Exception):
            vc.compute_contract_correctness_gate(descriptor, {"disposition": "passed"})


@dataclass
class _FakeVerdict:
    eligible: bool


class ComputePersistedValidationEligibleTests(unittest.TestCase):
    """GPT round 8 (req_84fca34f83064678): a bound-contract patch must
    never have the raw adapter verdict persisted as its evidence
    eligibility -- that would let patch-verify-evidence read adapter-only
    evidence as full contract qualification, which VA13 explicitly cannot
    grant (VA14's job)."""

    def test_bound_contract_forces_false_even_when_adapter_verdict_is_eligible(self) -> None:
        descriptor = _FakeDescriptor("RD08-Q6K-MMVQ-VDR2")
        result = vc.compute_persisted_validation_eligible(descriptor, _FakeVerdict(eligible=True))
        self.assertFalse(result)

    def test_bound_contract_forces_false_even_with_no_verdict_at_all(self) -> None:
        descriptor = _FakeDescriptor("RD08-Q6K-MMVQ-VDR2")
        self.assertFalse(vc.compute_persisted_validation_eligible(descriptor, None))

    def test_no_contract_passes_through_the_real_adapter_verdict(self) -> None:
        descriptor = _FakeDescriptor(None)
        self.assertTrue(vc.compute_persisted_validation_eligible(descriptor, _FakeVerdict(eligible=True)))
        self.assertFalse(vc.compute_persisted_validation_eligible(descriptor, _FakeVerdict(eligible=False)))

    def test_no_contract_no_verdict_returns_none(self) -> None:
        descriptor = _FakeDescriptor(None)
        self.assertIsNone(vc.compute_persisted_validation_eligible(descriptor, None))


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
