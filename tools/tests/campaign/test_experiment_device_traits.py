"""RD92 device-trait scope and eligibility tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.experiment import contract as ec  # noqa: E402


def _document(*, scope_overrides: dict[str, object] | None = None) -> dict[str, object]:
    scope: dict[str, object] = {
        "backend": "hip",
        "architectures": ["gfx1151"],
        "weight_types": ["q8_0"],
    }
    if scope_overrides:
        scope.update(scope_overrides)
    return {
        "title": "integrated host-buffer correctness contract",
        "source": {
            "source_id": "rd22-source",
            "commits": ["deadbeef"],
            "atomic_part": "rd22-host-buffer",
        },
        "hypothesis": {
            "family": "mmq",
            "expected_effect": "correctness",
            "rationale": "integrated host-buffer async correctness",
        },
        "prerequisites": [],
        "scope": scope,
        "positive": {"models": ["model"], "workloads": ["small_m"]},
        "controls": {"models": ["control"], "workloads": ["small_m"]},
        "boundary": {"dimensions": {}},
        "correctness": {"backend_reference": "required"},
        "acceptance": {"max_control_regression_pct": 1},
    }


class DeviceTraitEligibilityTests(unittest.TestCase):
    def test_matching_device_traits_are_eligible(self) -> None:
        contract = ec.parse_contract(
            _document(
                scope_overrides={
                    "integrated": True,
                    "uma": True,
                    "peer_access": True,
                    "gpu_count": {"minimum": 2},
                    "driver": {"minimum": [24, 20, 1]},
                }
            ),
            contract_id="RD22",
        )
        self.assertTrue(
            ec.evaluate_scope_eligibility(
                contract.scope,
                ec.DeviceTraits(
                    integrated=True,
                    uma=True,
                    peer_access=True,
                    gpu_count=2,
                    driver_version=ec.DriverVersion(24, 20, 2),
                ),
            )
        )
        self.assertIsInstance(contract.scope.gpu_count, ec.GpuCountConstraint)
        self.assertIsInstance(contract.scope.driver, ec.DriverVersionConstraint)

    def test_mismatched_device_trait_is_ineligible(self) -> None:
        contract = ec.parse_contract(
            _document(scope_overrides={"integrated": True, "uma": True}),
            contract_id="RD22",
        )
        self.assertFalse(
            ec.evaluate_scope_eligibility(
                contract.scope,
                ec.DeviceTraits(
                    integrated=False,
                    uma=False,
                    peer_access=True,
                    gpu_count=1,
                ),
            )
        )

    def test_unverifiable_driver_requirement_fails_closed(self) -> None:
        contract = ec.parse_contract(
            _document(scope_overrides={"driver": {"minimum": [24, 20, 1]}}),
            contract_id="RD22",
        )
        with self.assertRaisesRegex(ec.ExperimentContractError, "driver.*cannot be verified"):
            ec.evaluate_scope_eligibility(
                contract.scope,
                ec.DeviceTraits(
                    integrated=True,
                    uma=True,
                    peer_access=True,
                    gpu_count=1,
                ),
            )

    def test_device_trait_requirements_change_contract_identity(self) -> None:
        base = ec.parse_contract(_document(), contract_id="RD22")
        trait_scoped = ec.parse_contract(
            _document(scope_overrides={"integrated": True}),
            contract_id="RD22",
        )
        self.assertNotEqual(base.contract_hash, trait_scoped.contract_hash)


if __name__ == "__main__":
    unittest.main()
