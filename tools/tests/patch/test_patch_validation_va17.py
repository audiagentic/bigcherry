"""VA17 slice 1: contract-scoped ValidationPlan core semantics.

GPT review (session ses_5bbee8ce5c9a4265, req_e3f0e25329814273 /
req_4fdf8cdfa7d94e20 follow-ups): CheckSpec gains contract_ids (routing
metadata parsed from a generic adapter ``contracts = [...]`` field);
ValidationPlan.contract becomes plural ValidationPlan.contracts, with the
singular ``.contract`` kept ONLY as a 0/1-contract compatibility property
that fails closed for >1 bound contracts; requirements become
(contract_id, capability) pairs, independently proven per contract.

The real authoritative multi-contract fixture on this branch is
1203_rd050607_rdna4_wmma_fa_q6k_mmq, which binds
RD05-WMMA-FA-CORRECTNESS-BARRIERS, RD06-RDNA4-WMMA-FA-CONFIG, and
RD07-Q6K-MMQ-PREFILL-FOLD -- used here as the real descriptor/contract-set
fixture (it has no validation.toml yet, so it is NOT used as an
end-to-end plan fixture -- that is future, separate work). Plan-level
scoping/coverage scenarios are built from these three contracts' real
bind_contract() output plus synthetic CheckSpecs, per GPT's explicit
direction not to fabricate a fourth, made-up contract.

VALIDATION_FRAMEWORK_VERSION deliberately stays "2" in this slice (GPT:
bumping to 3 would stale existing VA04/VA05 evidence for no reason --
VA17 adds support for a previously fail-closed/unsupported domain, it
does not change existing 0/1-contract semantics).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.experiment import contract as experiment_contract  # noqa: E402
from bigcherry.patch import registry as patch_registry  # noqa: E402
from bigcherry.patch import validation as pv  # noqa: E402
from bigcherry.core import paths  # noqa: E402


def _real_bindings() -> tuple[pv.ContractBinding, pv.ContractBinding, pv.ContractBinding]:
    registry = experiment_contract.load_contracts(paths.EXPERIMENT_CONTRACTS)
    rd05 = pv.bind_contract(registry.contracts["RD05-WMMA-FA-CORRECTNESS-BARRIERS"])
    rd06 = pv.bind_contract(registry.contracts["RD06-RDNA4-WMMA-FA-CONFIG"])
    rd07 = pv.bind_contract(registry.contracts["RD07-Q6K-MMQ-PREFILL-FOLD"])
    return rd05, rd06, rd07


class RealDescriptorResolutionTests(unittest.TestCase):
    def test_real_1203_descriptor_resolves_all_three_real_bindings(self) -> None:
        reg = patch_registry.load_registry(paths.PATCHES)
        descriptor = next(
            d for d in reg.descriptors if d.patch_id == "1203_rd050607_rdna4_wmma_fa_q6k_mmq"
        )
        self.assertEqual(
            descriptor.experiment_contracts,
            (
                "RD05-WMMA-FA-CORRECTNESS-BARRIERS",
                "RD06-RDNA4-WMMA-FA-CONFIG",
                "RD07-Q6K-MMQ-PREFILL-FOLD",
            ),
        )
        contracts = pv.load_contracts_for_descriptor(descriptor)
        self.assertEqual(len(contracts), 3)
        self.assertEqual(
            {c.id for c in contracts},
            {
                "RD05-WMMA-FA-CORRECTNESS-BARRIERS",
                "RD06-RDNA4-WMMA-FA-CONFIG",
                "RD07-Q6K-MMQ-PREFILL-FOLD",
            },
        )

    def test_real_1203_descriptor_has_no_adapter_yet(self) -> None:
        # Confirms this fixture is a real descriptor/contract-set fixture
        # only -- not (yet) an end-to-end plan fixture.
        reg = patch_registry.load_registry(paths.PATCHES)
        descriptor = next(
            d for d in reg.descriptors if d.patch_id == "1203_rd050607_rdna4_wmma_fa_q6k_mmq"
        )
        self.assertIsNone(descriptor.validation_path)


class MultiContractCoverageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rd05, self.rd06, self.rd07 = _real_bindings()

    def _apply_build(self) -> tuple[pv.CheckSpec, ...]:
        return (
            pv.CheckSpec(check_id="apply", capability="apply", validator="apply", required=True),
            pv.CheckSpec(check_id="build", capability="build", validator="build", required=True),
        )

    def test_rd07_only_scoped_correctness_cannot_satisfy_rd05_rd06(self) -> None:
        checks = self._apply_build() + (
            pv.CheckSpec(
                check_id="correctness-rd07", capability="correctness", validator="backend-ops",
                required=True, contract_ids=("RD07-Q6K-MMQ-PREFILL-FOLD",),
            ),
        )
        with self.assertRaises(pv.ConfigurationError) as ctx:
            pv.build_validation_plan(
                "1203_test", checks, bindings=(self.rd05, self.rd06, self.rd07),
            )
        message = str(ctx.exception)
        self.assertIn("RD05-WMMA-FA-CORRECTNESS-BARRIERS:correctness", message)
        self.assertIn("RD06-RDNA4-WMMA-FA-CONFIG:correctness", message)

    def test_scoped_to_two_contracts_satisfies_both(self) -> None:
        checks = self._apply_build() + (
            pv.CheckSpec(
                check_id="correctness-rd05-rd06", capability="correctness", validator="backend-ops",
                required=True,
                contract_ids=("RD05-WMMA-FA-CORRECTNESS-BARRIERS", "RD06-RDNA4-WMMA-FA-CONFIG"),
            ),
            pv.CheckSpec(
                check_id="controls-rd05-rd06", capability="controls", validator="benchmark",
                required=True,
                contract_ids=("RD05-WMMA-FA-CORRECTNESS-BARRIERS", "RD06-RDNA4-WMMA-FA-CONFIG"),
            ),
            pv.CheckSpec(
                check_id="performance-rd06", capability="performance", validator="benchmark",
                required=True, contract_ids=("RD06-RDNA4-WMMA-FA-CONFIG",),
            ),
            pv.CheckSpec(
                check_id="activation-rd06", capability="activation", validator="trace-marker",
                required=True, contract_ids=("RD06-RDNA4-WMMA-FA-CONFIG",),
            ),
            pv.CheckSpec(
                check_id="correctness-rd07", capability="correctness", validator="backend-ops",
                required=True, contract_ids=("RD07-Q6K-MMQ-PREFILL-FOLD",),
            ),
            pv.CheckSpec(
                check_id="controls-rd07", capability="controls", validator="benchmark",
                required=True, contract_ids=("RD07-Q6K-MMQ-PREFILL-FOLD",),
            ),
            pv.CheckSpec(
                check_id="performance-rd07", capability="performance", validator="benchmark",
                required=True, contract_ids=("RD07-Q6K-MMQ-PREFILL-FOLD",),
            ),
            pv.CheckSpec(
                check_id="activation-rd07", capability="activation", validator="trace-marker",
                required=True, contract_ids=("RD07-Q6K-MMQ-PREFILL-FOLD",),
            ),
        )
        plan = pv.build_validation_plan(
            "1203_test", checks, bindings=(self.rd05, self.rd06, self.rd07),
        )
        self.assertEqual(len(plan.contracts), 3)
        self.assertIn(("RD05-WMMA-FA-CORRECTNESS-BARRIERS", "correctness"), plan.contract_requirements)
        self.assertIn(("RD06-RDNA4-WMMA-FA-CONFIG", "correctness"), plan.contract_requirements)

    def test_unscoped_check_on_multi_contract_patch_satisfies_none(self) -> None:
        checks = self._apply_build() + (
            pv.CheckSpec(
                check_id="correctness-unscoped", capability="correctness", validator="backend-ops",
                required=True,
            ),
        )
        with self.assertRaises(pv.ConfigurationError) as ctx:
            pv.build_validation_plan(
                "1203_test", checks, bindings=(self.rd05, self.rd06, self.rd07),
            )
        message = str(ctx.exception)
        # every contract's correctness requirement must be reported missing --
        # the unscoped check satisfied none of them.
        self.assertIn("RD05-WMMA-FA-CORRECTNESS-BARRIERS:correctness", message)
        self.assertIn("RD06-RDNA4-WMMA-FA-CONFIG:correctness", message)
        self.assertIn("RD07-Q6K-MMQ-PREFILL-FOLD:correctness", message)

    def test_unscoped_single_contract_behavior_is_unchanged(self) -> None:
        # 0/1-contract compat: an unscoped check keeps implicitly covering
        # the sole bound contract, exactly as before this slice.
        checks = self._apply_build() + (
            pv.CheckSpec(
                check_id="correctness", capability="correctness", validator="backend-ops",
                required=True,
            ),
            pv.CheckSpec(
                check_id="controls", capability="controls", validator="benchmark", required=True,
            ),
        )
        plan = pv.build_validation_plan("1203_test", checks, binding=self.rd05)
        self.assertEqual(plan.contract.contract_id, "RD05-WMMA-FA-CORRECTNESS-BARRIERS")
        self.assertEqual(len(plan.contracts), 1)

    def test_unknown_check_contract_id_rejected(self) -> None:
        checks = self._apply_build() + (
            pv.CheckSpec(
                check_id="correctness-bogus", capability="correctness", validator="backend-ops",
                required=True, contract_ids=("RD99-DOES-NOT-EXIST",),
            ),
        )
        with self.assertRaises(pv.ConfigurationError) as ctx:
            pv.build_validation_plan("1203_test", checks, bindings=(self.rd05,))
        self.assertIn("RD99-DOES-NOT-EXIST", str(ctx.exception))

    def test_contract_compatibility_property_fails_closed_for_multiple(self) -> None:
        checks = self._apply_build() + (
            pv.CheckSpec(
                check_id="correctness-rd05-rd06", capability="correctness", validator="backend-ops",
                required=True,
                contract_ids=("RD05-WMMA-FA-CORRECTNESS-BARRIERS", "RD06-RDNA4-WMMA-FA-CONFIG"),
            ),
        )
        # Deliberately incomplete plan for just RD05/RD06 (no RD07 producers)
        # -- constructed directly to test the .contract property in
        # isolation without needing a fully valid multi-contract plan.
        plan = pv.ValidationPlan(
            patch_id="1203_test", checks=checks, universal_capabilities=pv.UNIVERSAL_REQUIREMENTS,
            contracts=(self.rd05, self.rd06),
        )
        with self.assertRaises(pv.ConfigurationError):
            _ = plan.contract

    def test_binding_and_bindings_kwargs_are_mutually_exclusive(self) -> None:
        with self.assertRaises(pv.ConfigurationError):
            pv.build_validation_plan(
                "1203_test", self._apply_build(), binding=self.rd05, bindings=(self.rd06,),
            )


class ContractsToyParsingTests(unittest.TestCase):
    def _write(self, tmp_path: Path, body: str) -> Path:
        path = tmp_path / "validation.toml"
        path.write_text(body, encoding="utf-8")
        return path

    def test_contracts_field_parses_into_contract_ids(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(
                Path(tmp),
                """
schema = 1
[[check]]
id = "correctness-rd05-rd06"
capability = "correctness"
validator = "backend-ops"
required = true
contracts = ["RD05-WMMA-FA-CORRECTNESS-BARRIERS", "RD06-RDNA4-WMMA-FA-CONFIG"]
ops = ["SOME_OP"]
""",
            )
            specs = pv.parse_validation_toml(path, patch_id="1203_test")
            self.assertEqual(len(specs), 1)
            self.assertEqual(
                specs[0].contract_ids,
                ("RD05-WMMA-FA-CORRECTNESS-BARRIERS", "RD06-RDNA4-WMMA-FA-CONFIG"),
            )
            # 'contracts' must never leak into validator config.
            self.assertNotIn("contracts", specs[0].config)
            self.assertEqual(specs[0].config, {"ops": ["SOME_OP"]})

    def test_empty_contracts_list_rejected(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(
                Path(tmp),
                """
schema = 1
[[check]]
id = "x"
capability = "apply"
validator = "apply"
required = true
contracts = []
""",
            )
            with self.assertRaises(pv.ConfigurationError):
                pv.parse_validation_toml(path, patch_id="1203_test")

    def test_duplicate_contracts_list_rejected(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(
                Path(tmp),
                """
schema = 1
[[check]]
id = "x"
capability = "apply"
validator = "apply"
required = true
contracts = ["RD05-WMMA-FA-CORRECTNESS-BARRIERS", "RD05-WMMA-FA-CORRECTNESS-BARRIERS"]
""",
            )
            with self.assertRaises(pv.ConfigurationError):
                pv.parse_validation_toml(path, patch_id="1203_test")


class PlanDigestV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.rd05, self.rd06 = _real_bindings()[:2]

    def _checks(self, extra_scope: tuple[str, ...]) -> tuple[pv.CheckSpec, ...]:
        # Fixed checks that fully satisfy RD05's and RD06's real
        # requirements on their own, regardless of `extra_scope` --
        # `extra-redundant` is an ADDITIONAL, coverage-irrelevant check
        # (correctness is already satisfied) whose own contract scope is
        # the only thing that varies between callers, isolating exactly
        # the scope-identity change the v2 payload/digest must be
        # sensitive to.
        rd05_id, rd06_id = self.rd05.contract_id, self.rd06.contract_id
        return (
            pv.CheckSpec(check_id="apply", capability="apply", validator="apply", required=True),
            pv.CheckSpec(check_id="build", capability="build", validator="build", required=True),
            pv.CheckSpec(
                check_id="correctness-rd05", capability="correctness", validator="backend-ops",
                required=True, contract_ids=(rd05_id,),
            ),
            pv.CheckSpec(
                check_id="controls-rd05", capability="controls", validator="benchmark",
                required=True, contract_ids=(rd05_id,),
            ),
            pv.CheckSpec(
                check_id="correctness-rd06", capability="correctness", validator="backend-ops",
                required=True, contract_ids=(rd06_id,),
            ),
            pv.CheckSpec(
                check_id="controls-rd06", capability="controls", validator="benchmark",
                required=True, contract_ids=(rd06_id,),
            ),
            pv.CheckSpec(
                check_id="performance-rd06", capability="performance", validator="benchmark",
                required=True, contract_ids=(rd06_id,),
            ),
            pv.CheckSpec(
                check_id="activation-rd06", capability="activation", validator="trace-marker",
                required=True, contract_ids=(rd06_id,),
            ),
            pv.CheckSpec(
                check_id="extra-redundant", capability="correctness", validator="backend-ops",
                required=False, contract_ids=extra_scope,
            ),
        )

    def test_plan_digest_changes_when_only_contract_scope_changes(self) -> None:
        rd05_id, rd06_id = self.rd05.contract_id, self.rd06.contract_id
        plan_a = pv.build_validation_plan(
            "1203_test", self._checks((rd05_id,)), bindings=(self.rd05, self.rd06),
        )
        # Same capabilities, same check ids, same coverage outcome -- only
        # the redundant extra check's own declared contract scope changes.
        plan_b = pv.build_validation_plan(
            "1203_test", self._checks((rd06_id,)), bindings=(self.rd05, self.rd06),
        )
        digest_a = pv.plan_digest(plan_a)
        digest_b = pv.plan_digest(plan_b)
        self.assertNotEqual(digest_a, digest_b)

    def test_v2_schema_and_deterministic_ordering(self) -> None:
        rd05_id, rd06_id = self.rd05.contract_id, self.rd06.contract_id
        plan = pv.build_validation_plan(
            "1203_test", self._checks((rd05_id,)),
            bindings=(self.rd06, self.rd05),  # deliberately reversed input order
        )
        payload = pv.plan_canonical_payload(plan)
        self.assertEqual(payload["schema"], "bigcherry-validation-plan/v2")
        self.assertEqual(
            [entry["id"] for entry in payload["contracts"]],
            sorted([rd05_id, rd06_id]),
        )
        self.assertEqual(payload["contract_requirements"], sorted(payload["contract_requirements"]))
        # Rebuilding with the bindings in the OTHER order must serialize
        # identically (order-independent, deterministic).
        plan2 = pv.build_validation_plan(
            "1203_test", self._checks((rd05_id,)), bindings=(self.rd05, self.rd06),
        )
        self.assertEqual(pv.plan_canonical_payload(plan), pv.plan_canonical_payload(plan2))


if __name__ == "__main__":
    unittest.main()
