"""Real framework adapters cannot qualify from fixture checks alone."""

from pathlib import Path

import pytest

from bigcherry.patch import registry, validation, validation_policy


ROOT = Path(__file__).resolve().parents[3] / "patches"


@pytest.mark.parametrize("patch_id,check_id", [
    ("0100_cmake_options", "coverage-source-selection"),
    ("0700_coverage_counters", "family-hook-isolation"),
])
def test_framework_fixture_success_does_not_supply_apply_or_hip_build(patch_id, check_id):
    descriptor = registry.load_registry(ROOT).get(patch_id)
    plan = validation_policy.require_execution_package(descriptor, root=ROOT)
    assert plan.contracts == ()
    assert {check.check_id for check in plan.checks} == {"apply", "build", check_id}
    assert all(check.required for check in plan.checks)
    fixture_result = validation.ValidationResult(
        check_id, "configuration", validation.PASS, "host fixture only")
    verdict = validation.compute_verdict(plan, {check_id: fixture_result})
    assert not verdict.eligible
    assert {result.check_id for result in verdict.results if result.status == validation.ERROR} == {
        "apply", "build"}
