"""Real framework adapters cannot qualify from fixture checks alone."""

from pathlib import Path
import tempfile

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


@pytest.mark.parametrize("patch_id", ["0100_cmake_options", "0700_coverage_counters"])
def test_framework_apply_and_build_are_package_custom_producers(patch_id):
    descriptor = registry.load_registry(ROOT).get(patch_id)
    plan = validation_policy.require_execution_package(descriptor, root=ROOT)
    checks = {check.check_id: check for check in plan.checks}
    assert checks["apply"].validator == "custom"
    assert checks["build"].validator == "custom"
    assert checks["apply"].config["callable"].endswith("validation/checks.py:check_apply")
    assert checks["build"].config["callable"].endswith("validation/checks.py:check_build")
    ctx = validation.ValidationContext(
        descriptor=descriptor,
        base_revision="test",
        control_source=None,
        subject_source=None,
        package_root=ROOT / patch_id,
    )
    assert validation.evaluate_check(checks["apply"], ctx).status == validation.BLOCKED
    assert validation.evaluate_check(checks["build"], ctx).status == validation.BLOCKED


@pytest.mark.parametrize("patch_id", ["0100_cmake_options", "0700_coverage_counters"])
def test_framework_custom_producers_pass_single_composition_with_bound_builds(patch_id):
    descriptor = registry.load_registry(ROOT).get(patch_id)
    plan = validation_policy.require_execution_package(descriptor, root=ROOT)
    checks = {check.check_id: check for check in plan.checks}
    with tempfile.TemporaryDirectory() as directory:
        run_dir = Path(directory)
        register = validation.make_default_register_artifact(run_dir)
        apply_file = run_dir / "apply.json"
        apply_file.write_text('{"composition":["framework"]}', encoding="utf-8")
        production_file = run_dir / "production.json"
        production_file.write_text('{"build":"production"}', encoding="utf-8")
        diagnostic_file = run_dir / "diagnostic.json"
        diagnostic_file.write_text('{"build":"diagnostic"}', encoding="utf-8")
        apply_ref = register("apply.json", apply_file)
        production_ref = register("production.json", production_file)
        diagnostic_ref = register("diagnostic.json", diagnostic_file)
        ctx = validation.ValidationContext(
            descriptor=descriptor, base_revision="test", control_source=None,
            subject_source=None, package_root=ROOT / patch_id, run_dir=run_dir,
            configuration_evidence={
                "apply": {"single_composition": True, "verified": True,
                           "idempotent": True, "artifact": apply_ref.__dict__},
                "builds": {
                    "production": {"completed": True, "artifact": production_ref.__dict__},
                    "diagnostic": {"completed": True, "artifact": diagnostic_ref.__dict__},
                },
            },
        )
        assert validation.evaluate_check(checks["apply"], ctx).status == validation.PASS
        assert validation.evaluate_check(checks["build"], ctx).status == validation.PASS


@pytest.mark.parametrize("patch_id", ["0100_cmake_options", "0700_coverage_counters"])
def test_framework_custom_producers_reject_stale_or_tampered_single_composition_proof(patch_id):
    descriptor = registry.load_registry(ROOT).get(patch_id)
    plan = validation_policy.require_execution_package(descriptor, root=ROOT)
    apply_spec = next(check for check in plan.checks if check.check_id == "apply")
    with tempfile.TemporaryDirectory() as directory:
        run_dir = Path(directory)
        register = validation.make_default_register_artifact(run_dir)
        proof = run_dir / "apply.json"
        proof.write_text("proof", encoding="utf-8")
        ref = register("apply.json", proof)
        (run_dir / ref.path).write_text("tampered", encoding="utf-8")
        ctx = validation.ValidationContext(
            descriptor=descriptor, base_revision="test", control_source=None,
            subject_source=None, package_root=ROOT / patch_id, run_dir=run_dir,
            configuration_evidence={"apply": {
                "single_composition": True, "verified": True, "idempotent": True,
                "artifact": ref.__dict__,
            }},
        )
        result = validation.evaluate_check(apply_spec, ctx)
        assert result.status == validation.BLOCKED
