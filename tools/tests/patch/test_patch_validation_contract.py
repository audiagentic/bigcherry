"""RS08 tests: Experiment Contract binding into the validation plan
(patch-system PA03; runbook section 61).

Verification list per the runbook:
  - contract exists
  - contract hash captured
  - scope/backend compatible with patch metadata
  - declared hardware/architecture not contradictory
  - correctness requirements get required producers
  - performance acceptance requires performance evidence
  - controls/boundaries appear in validation plan
  - no Experiment Contract fields duplicated in validation.toml

Exit criterion: changing ONLY the Experiment Contract changes validation
identity but NOT implementation/source identity.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.experiment import contract as experiment_contract # noqa: E402
from bigcherry.patch import registry as patch_registry # noqa: E402
from bigcherry.patch import source as psi # noqa: E402
from bigcherry.patch import validation as pv # noqa: E402

CONTRACT_TOML = """\
[contract.ec-test]
title = "test contract"

[contract.ec-test.source]
source_id = "stew675-rdna-boosts"
commits = ["abc123def456"]
atomic_part = "1201_test"

[contract.ec-test.hypothesis]
family = "mmvq"
expected_effect = "{effect}"
rationale = "test hypothesis"

[contract.ec-test.scope]
backend = "{backend}"
architectures = ["gfx1100"]
weight_types = ["q6_k"]

[contract.ec-test.positive]
models = ["m1"]
workloads = ["decode"]

[contract.ec-test.controls]
models = ["m1"]
workloads = ["prefill"]

[contract.ec-test.correctness]
bit_identical = "required"

[contract.ec-test.acceptance]
target_kernel_gain_pct = {gain}
max_control_regression_pct = 1
"""

PATCH_TOML = """\
schema = 1
id = "1201_test"
order = 1201
group = "test"
state = "untested"
kind = "framework"
origin = "local"
backend = "hip"
experiment-contract = "ec-test"
{extra}
"""

PATCH_PY = "PATCHES = []\n"

ADAPTER_FULL = """\
schema = 1

[[check]]
id = "applies"
capability = "apply"
validator = "apply"
required = true

[[check]]
id = "builds"
capability = "build"
validator = "build"
required = true

[[check]]
id = "correct"
capability = "correctness"
validator = "backend-ops"
required = true
ops = ["MUL_MAT"]

[[check]]
id = "bench"
capability = "performance"
validator = "benchmark"
required = true

[[check]]
id = "control"
capability = "controls"
validator = "benchmark"
required = true

[[check]]
id = "activate"
capability = "activation"
validator = "trace-marker"
required = true
marker-regex = "BIGCHERRY_PATCH_HIT"
"""


class ContractBindingTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="bigcherry-pvc-")
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name)
        self.root = self.base / "patches" / "rd" / "1201_test"
        self.root.mkdir(parents=True)
        self.contracts = self.base / "contracts.toml"
        self.write_tree(adapter=ADAPTER_FULL, effect="both", gain=0.3)

    def write_tree(self, *, adapter: str | None, effect: str = "both", gain=0.3,
                   contract: str | None = None, patch_toml_extra: str = "") -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        body = contract if contract is not None else CONTRACT_TOML.format(
            effect=effect, backend="hip", gain=gain
        )
        self.contracts.write_text(body, encoding="utf-8")
        (self.root / "patch.toml").write_text(
            PATCH_TOML.format(extra=patch_toml_extra), encoding="utf-8"
        )
        (self.root / "patch.py").write_text(PATCH_PY, encoding="utf-8")
        if adapter is not None:
            (self.root / "validation.toml").write_text(adapter, encoding="utf-8")
        else:
            (self.root / "validation.toml").unlink(missing_ok=True)

    def _registry(self) -> patch_registry.PatchRegistry:
        return patch_registry.load_registry(self.root.parent.parent,
                                            contracts_path=self.contracts)

    def _plan(self) -> pv.ValidationPlan | None:
        registry = self._registry()
        descriptor = registry.get("1201_test")
        return pv.build_plan_for_patch(
            descriptor, root=self.root.parent.parent, contracts_path=self.contracts
        )

    def test_plan_captures_contract_hash(self) -> None:
        registry = self._registry()
        descriptor = registry.get("1201_test")
        plan = self._plan()
        self.assertIsNotNone(plan)
        contract = experiment_contract.load_contracts(self.contracts)["ec-test"]
        self.assertEqual(plan.contract.contract_hash, contract.contract_hash)
        self.assertEqual(plan.contract.contract_id, "ec-test")

    def test_scope_backend_compatible(self) -> None:
        self.assertIsNotNone(self._plan())

    def test_backend_contradiction_is_configuration_error(self) -> None:
        self.write_tree(adapter=ADAPTER_FULL,
                        contract=CONTRACT_TOML.format(effect="both", backend="cuda", gain=0.3))
        with self.assertRaisesRegex(pv.ConfigurationError, "backend"):
            self._plan()

    def test_architecture_contradiction_is_configuration_error(self) -> None:
        self.write_tree(
            adapter=ADAPTER_FULL,
            patch_toml_extra='validation-architectures = ["gfx1101"]\n',
        )
        with self.assertRaisesRegex(pv.ConfigurationError, "architectures"):
            self._plan()

    def test_correctness_requires_producer(self) -> None:
        # Adapter proves apply/build only; the contract demands correctness.
        self.write_tree(
            adapter="schema = 1\n\n"
            "[[check]]\nid = \"applies\"\ncapability = \"apply\"\nvalidator = \"apply\"\n"
            "[[check]]\nid = \"builds\"\ncapability = \"build\"\nvalidator = \"build\"\n"
        )
        with self.assertRaisesRegex(pv.ConfigurationError, "correctness"):
            self._plan()

    def test_correctness_only_contract_with_gain_requires_performance(self) -> None:
        # A correctness hypothesis still needs a performance producer when its
        # acceptance section declares a gain threshold.
        self.write_tree(
            effect="correctness",
            adapter="schema = 1\n\n"
            "[[check]]\nid = \"applies\"\ncapability = \"apply\"\nvalidator = \"apply\"\n"
            "[[check]]\nid = \"builds\"\ncapability = \"build\"\nvalidator = \"build\"\n"
            "[[check]]\nid = \"correct\"\ncapability = \"correctness\"\nvalidator = \"backend-ops\"\n",
        )
        with self.assertRaisesRegex(pv.ConfigurationError, "performance"):
            self._plan()

    def test_performance_acceptance_requires_performance_evidence(self) -> None:
        # expected_effect = performance; adapter has no benchmark/autotune.
        self.write_tree(
            adapter="schema = 1\n\n"
            "[[check]]\nid = \"applies\"\ncapability = \"apply\"\nvalidator = \"apply\"\n"
            "[[check]]\nid = \"builds\"\ncapability = \"build\"\nvalidator = \"build\"\n"
            "[[check]]\nid = \"correct\"\ncapability = \"correctness\"\nvalidator = \"backend-ops\"\n"
            "[[check]]\nid = \"activate\"\ncapability = \"activation\"\nvalidator = \"trace-marker\"\n"
            "[[check]]\nid = \"control\"\ncapability = \"controls\"\nvalidator = \"benchmark\"\n"
            .replace("[[check]]\nid = \"activate\"", "[[check]]\nid = \"activate\"\nvalidator = \"trace-marker\"\n", 1)
            if False else
            "schema = 1\n\n"
            "[[check]]\nid = \"applies\"\ncapability = \"apply\"\nvalidator = \"apply\"\n"
            "[[check]]\nid = \"builds\"\ncapability = \"build\"\nvalidator = \"build\"\n"
            "[[check]]\nid = \"correct\"\ncapability = \"correctness\"\nvalidator = \"backend-ops\"\n"
        )
        with self.assertRaisesRegex(pv.ConfigurationError, "performance"):
            self._plan()

    def test_controls_boundaries_require_producer(self) -> None:
        # Contract has non-empty controls; adapter has no controls producer.
        self.write_tree(
            adapter="schema = 1\n\n"
            "[[check]]\nid = \"applies\"\ncapability = \"apply\"\nvalidator = \"apply\"\n"
            "[[check]]\nid = \"builds\"\ncapability = \"build\"\nvalidator = \"build\"\n"
            "[[check]]\nid = \"correct\"\ncapability = \"correctness\"\nvalidator = \"backend-ops\"\n"
            "[[check]]\nid = \"bench\"\ncapability = \"performance\"\nvalidator = \"benchmark\"\n"
            "[[check]]\nid = \"activate\"\ncapability = \"activation\"\nvalidator = \"trace-marker\"\n"
        )
        with self.assertRaisesRegex(pv.ConfigurationError, "controls"):
            self._plan()

    def test_full_plan_builds_with_all_producers(self) -> None:
        plan = self._plan()
        self.assertIsNotNone(plan)
        for capability in ("apply", "build", "performance", "activation",
                           "correctness", "controls"):
            self.assertIn(capability, plan.required_capabilities)
        self.assertEqual(plan.contract.contract_id, "ec-test")

    def test_nested_contract_fields_not_allowed_in_adapter(self) -> None:
        self.write_tree(
            adapter=(
                "schema = 1\n\n"
                "[[check]]\n"
                "id = \"applies\"\ncapability = \"apply\"\nvalidator = \"apply\"\n"
                "config = { boundary = { source_id = \"smuggled\" } }\n"
            ),
        )
        with self.assertRaisesRegex(pv.ConfigurationError, "Experiment Contract"):
            self._plan()

    def test_contract_fields_not_allowed_in_adapter(self) -> None:
        self.write_tree(
            adapter="schema = 1\n\n"
            "[[check]]\nid = \"applies\"\ncapability = \"apply\"\nvalidator = \"apply\"\n"
            "end_to_end_gain_pct = 2.5\n"
        )
        with self.assertRaisesRegex(pv.ConfigurationError, "Experiment Contract"):
            self._plan()

    def test_contract_without_adapter_fails_closed(self) -> None:
        self.write_tree(adapter=None)
        with self.assertRaisesRegex(pv.ConfigurationError, "no validation.toml"):
            self._plan()

    def test_legacy_flat_patch_has_no_plan(self) -> None:
        # A flat patch with no contract and no adapter is simply not under
        # validation: None, not an error.
        flat_root = self.base / "patches"
        flat = flat_root / "0900_legacy.py"
        flat.write_text('STATE = "untested"\nPATCHES = []\n', encoding="utf-8")
        registry = patch_registry.load_registry(flat_root, contracts_path=self.contracts)
        descriptor = registry.get("0900_legacy")
        self.assertIsNone(
            pv.build_plan_for_patch(descriptor, root=flat_root,
                                    contracts_path=self.contracts)
        )

    def test_missing_contract_is_tree_error(self) -> None:
        # Discovery itself fails closed: patch.toml references a contract
        # that does not exist in the contract file.
        self.write_tree(adapter=ADAPTER_FULL,
                        contract=CONTRACT_TOML.format(effect="both", backend="hip", gain=0.3)
                        .replace("contract.ec-test", "contract.ec-other"))
        with self.assertRaises(patch_registry.PatchRegistryError):
            self._registry()


class ContractExitCriterionTests(unittest.TestCase):
    """Runbook RS08 exit criterion: changing ONLY the Experiment Contract
    changes validation identity but not implementation/source identity."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="bigcherry-pvc-exit-")
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name)
        self.patches_root = self.base / "patches"
        self.root = self.patches_root / "rd" / "1201_test"
        self.root.mkdir(parents=True)
        self.contracts = self.base / "contracts.toml"
        self.write()

    def write(self, gain=0.3) -> None:
        self.contracts.write_text(
            CONTRACT_TOML.format(effect="both", backend="hip", gain=gain),
            encoding="utf-8",
        )
        (self.root / "patch.toml").write_text(PATCH_TOML.format(extra=""),
                                              encoding="utf-8")
        (self.root / "patch.py").write_text(PATCH_PY, encoding="utf-8")
        (self.root / "validation.toml").write_text(ADAPTER_FULL, encoding="utf-8")

    def test_contract_change_changes_only_validation_identity(self) -> None:
        first = patch_registry.load_registry(self.patches_root,
                                             contracts_path=self.contracts).get("1201_test")
        self.write(gain=7.5)  # ONLY the contract changes
        second = patch_registry.load_registry(self.patches_root,
                                              contracts_path=self.contracts).get("1201_test")

        self.assertNotEqual(first.validation_digest, second.validation_digest,
                            "validation identity must change")
        self.assertEqual(first.implementation_digest, second.implementation_digest,
                         "implementation identity must NOT change")
        # Source identity (RS05/PA02 v2) is derived from the resolved base
        # revision + overlay digest + ordered implementation composition; the
        # contract-only change must not alter it.
        first_source_key = psi._make_source_identity_v2(
            resolved_revision="deadbeef",
            composition=(("1201_test", first.implementation_digest),),
            overlay_root=None,
        )["source_key"]
        second_source_key = psi._make_source_identity_v2(
            resolved_revision="deadbeef",
            composition=(("1201_test", second.implementation_digest),),
            overlay_root=None,
        )["source_key"]
        self.assertEqual(first_source_key, second_source_key,
                         "source identity must NOT change")
        # Build the first plan while the first contract is still current, then
        # change only the contract and build the second plan.
        self.write(gain=0.3)
        first_descriptor = patch_registry.load_registry(
            self.patches_root, contracts_path=self.contracts
        ).get("1201_test")
        plan_first = pv.build_plan_for_patch(
            first_descriptor, root=self.patches_root, contracts_path=self.contracts)
        self.write(gain=7.5)
        second_descriptor = patch_registry.load_registry(
            self.patches_root, contracts_path=self.contracts
        ).get("1201_test")
        plan_second = pv.build_plan_for_patch(
            second_descriptor, root=self.patches_root, contracts_path=self.contracts)
        self.assertNotEqual(pv.plan_digest(plan_first), pv.plan_digest(plan_second))


if __name__ == "__main__":
    unittest.main()
