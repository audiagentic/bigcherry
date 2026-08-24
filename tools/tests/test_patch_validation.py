"""RS07 tests for tools/bigcherry/patch_validation.py (patch-system PA03).

Runbook section 60 required negative tests:
  - unknown validator                      -> ERROR
  - missing required capability producer   -> configuration failure
  - custom callable escaping package       -> configuration failure
  - duplicate check ID                     -> configuration failure
  - required check blocked                 -> not eligible
  - required check error                   -> not eligible
plus the v1 lock (section 19): not_applicable never satisfies a required
check, and B4's framework-version re-export.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bigcherry import patch_registry  # noqa: E402
from bigcherry import patch_validation as pv  # noqa: E402

VALID_TOML = """\
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
"""

ADVISORY_ONLY_TOML = """\
schema = 1

[[check]]
id = "smoke"
capability = "smoke"
validator = "runtime-smoke"
required = false
"""


def _write(directory: Path, name: str, body: str) -> Path:
    path = directory / name
    path.write_text(body, encoding="utf-8")
    return path


def _descriptor(patch_id: str = "1201_test") -> patch_registry.PatchDescriptor:
    return patch_registry.PatchDescriptor(
        patch_id=patch_id,
        order=1201,
        representation=patch_registry.REPRESENTATION_SIMPLE,
        implementation_path=Path(f"{patch_id}.py"),
        package_root=None,
        metadata_path=None,
        group="test",
        state="untested",
        kind="experiment",
        origin="local",
        backend="hip",
        upstream=None,
        external_source=None,
        plan_ids=(),
        requires=(),
        conflicts=(),
        requires_options=(),
        forbids_options=(),
        subsystems=(),
        hardware=(),
        validation_architectures=(),
        experiment_contract=None,
        implementation_digest="d" * 64,
        validation_path=None,
        validation_digest=None,
    )


class ParserTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="bigcherry-pv-")
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def test_valid_file(self) -> None:
        path = _write(self.root, "validation.toml", VALID_TOML)
        specs = pv.parse_validation_toml(path, patch_id="1201_test")
        self.assertEqual(len(specs), 2)
        self.assertEqual(specs[0].check_id, "applies")
        self.assertEqual(specs[0].capability, "apply")
        self.assertEqual(specs[0].validator, "apply")
        self.assertTrue(specs[0].required)
        self.assertEqual(specs[0].config, {})

    def test_schema_must_be_one(self) -> None:
        path = _write(self.root, "validation.toml", VALID_TOML.replace("schema = 1", "schema = 2"))
        with self.assertRaises(pv.ConfigurationError):
            pv.parse_validation_toml(path)

    def test_missing_schema(self) -> None:
        path = _write(self.root, "validation.toml", VALID_TOML.replace("schema = 1\n\n", ""))
        with self.assertRaises(pv.ConfigurationError):
            pv.parse_validation_toml(path)

    def test_unknown_top_level_key(self) -> None:
        path = _write(self.root, "validation.toml", "hypothesis = 'faster'\n" + VALID_TOML)
        with self.assertRaisesRegex(pv.ConfigurationError, "unknown top-level"):
            pv.parse_validation_toml(path)

    def test_duplicate_check_id_is_configuration_failure(self) -> None:
        body = VALID_TOML + "\n[[check]]\nid = \"applies\"\ncapability = \"build\"\nvalidator = \"build\"\n"
        path = _write(self.root, "validation.toml", body)
        with self.assertRaisesRegex(pv.ConfigurationError, "duplicate check id"):
            pv.parse_validation_toml(path)

    def test_missing_required_key(self) -> None:
        path = _write(
            self.root,
            "validation.toml",
            "schema = 1\n\n[[check]]\nid = \"x\"\nvalidator = \"apply\"\n",
        )
        with self.assertRaisesRegex(pv.ConfigurationError, "capability"):
            pv.parse_validation_toml(path)

    def test_required_defaults_true(self) -> None:
        path = _write(
            self.root,
            "validation.toml",
            "schema = 1\n\n[[check]]\nid = \"x\"\ncapability = \"apply\"\nvalidator = \"apply\"\n",
        )
        specs = pv.parse_validation_toml(path)
        self.assertTrue(specs[0].required, "fail-closed default: checks are required")

    def test_validator_specific_config_passes_through(self) -> None:
        path = _write(
            self.root,
            "validation.toml",
            "schema = 1\n\n[[check]]\n"
            'id = "marker"\n'
            'capability = "activation"\n'
            'validator = "trace-marker"\n'
            'required = true\n'
            'marker-regex = "BIGCHERRY_PATCH_HIT"\n'
            "\n[check.negative-control.environment]\n"
            'GGML_DISABLE = "1"\n',
        )
        specs = pv.parse_validation_toml(path)
        self.assertEqual(specs[0].config["marker-regex"], "BIGCHERRY_PATCH_HIT")
        self.assertEqual(specs[0].config["negative-control"], {"environment": {"GGML_DISABLE": "1"}})


class PlanAggregationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="bigcherry-pv-")
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def _specs(self, body: str) -> tuple[pv.CheckSpec, ...]:
        return pv.parse_validation_toml(_write(self.root, "validation.toml", body))

    def test_missing_required_capability_producer_is_configuration_failure(self) -> None:
        # Universal requirements demand 'apply' and 'build'; an adapter
        # that only offers smoke must be a configuration error, not a skip.
        specs = self._specs(ADVISORY_ONLY_TOML)
        with self.assertRaisesRegex(pv.ConfigurationError, "no producer"):
            pv.build_validation_plan("1201_test", specs)

    def test_full_plan_builds(self) -> None:
        specs = self._specs(VALID_TOML)
        plan = pv.build_validation_plan("1201_test", specs)
        self.assertEqual(plan.required_capabilities, ("apply", "build"))
        self.assertIsNone(plan.contract)
        self.assertEqual(len(plan.checks), 2)
        self.assertEqual(len(plan.required_checks()), 2)

    def test_contract_requirements_aggregate(self) -> None:
        # A performance contract adds 'performance' + 'activation'; the
        # adapter must produce both or the plan fails closed.
        binding = pv.ContractBinding(
            contract_id="ec-test",
            contract_hash="h" * 32,
            expected_effect="performance",
            backend="hip",
            architectures=("gfx1100",),
            correctness_checks=("bit_identical",),
            has_controls=True,
            has_boundaries=False,
            required_capabilities=("performance", "activation", "correctness", "controls"),
        )
        specs = self._specs(VALID_TOML)  # only apply+build
        with self.assertRaisesRegex(pv.ConfigurationError, "performance"):
            pv.build_validation_plan("1201_test", specs, binding=binding)

        with open(self.root / "validation2.toml", "w", encoding="utf-8") as fh:
            fh.write(
                "schema = 1\n\n"
                "[[check]]\nid = \"applies\"\ncapability = \"apply\"\nvalidator = \"apply\"\n"
                "[[check]]\nid = \"builds\"\ncapability = \"build\"\nvalidator = \"build\"\n"
                '[[check]]\nid = "bench"\ncapability = "performance"\nvalidator = "benchmark"\n'
                '[[check]]\nid = "activate"\ncapability = "activation"\nvalidator = "trace-marker"\n'
                '[[check]]\nid = "correct"\ncapability = "correctness"\nvalidator = "backend-ops"\n'
                '[[check]]\nid = "control"\ncapability = "controls"\nvalidator = "benchmark"\n'
            )
        specs = pv.parse_validation_toml(self.root / "validation2.toml")
        plan = pv.build_validation_plan("1201_test", specs, binding=binding)
        self.assertIn("performance", plan.required_capabilities)
        self.assertIn("activation", plan.required_capabilities)
        self.assertIn("correctness", plan.required_capabilities)
        self.assertIn("controls", plan.required_capabilities)
        self.assertEqual(plan.contract.contract_id, "ec-test")

    def test_custom_check_can_produce_declared_capability(self) -> None:
        # A custom check satisfies the capability it declares (section 31),
        # so a plan with a custom 'activation' producer is legal.
        body = (
            "schema = 1\n\n"
            "[[check]]\nid = \"applies\"\ncapability = \"apply\"\nvalidator = \"apply\"\n"
            "[[check]]\nid = \"builds\"\ncapability = \"build\"\nvalidator = \"build\"\n"
            '[[check]]\nid = "act"\ncapability = "activation"\nvalidator = "custom"\nrequired = false\n'
            'callable = "validation/checks.py:act_check"\n'
        )
        specs = self._specs(body)
        plan = pv.build_validation_plan("1201_test", specs)
        self.assertIn("activation", [s.capability for s in plan.checks])


class CustomCallableTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="bigcherry-pv-custom-")
        self.addCleanup(self._tmp.cleanup)
        self.pkg = Path(self._tmp.name) / "pkg"
        (self.pkg / "validation").mkdir(parents=True)
        self.checks = self.pkg / "validation" / "checks.py"
        self.checks.write_text(
            "from bigcherry import patch_validation as pv\n"
            "def act_check(ctx):\n"
            "    return pv.ValidationResult(\n"
            "        check_id='act', capability='activation', status='pass', summary='ok')\n",
            encoding="utf-8",
        )

    def test_escaping_package_is_configuration_failure(self) -> None:
        with self.assertRaisesRegex(pv.ConfigurationError, "escapes the package root"):
            pv.resolve_custom_callable("../outside.py:fn", package_root=self.pkg)
        with self.assertRaisesRegex(pv.ConfigurationError, "escapes the package root"):
            pv.resolve_custom_callable("validation/../../outside.py:fn", package_root=self.pkg)
        with self.assertRaisesRegex(pv.ConfigurationError, "must be 'path"):
            pv.resolve_custom_callable("../etc/passwd:fn", package_root=self.pkg)

    def test_missing_file(self) -> None:
        with self.assertRaisesRegex(pv.ConfigurationError, "no file"):
            pv.resolve_custom_callable("validation/missing.py:fn", package_root=self.pkg)

    def test_missing_attribute(self) -> None:
        with self.assertRaisesRegex(pv.ConfigurationError, "no attribute"):
            pv.resolve_custom_callable("validation/checks.py:nope", package_root=self.pkg)

    def test_bad_signature_two_args(self) -> None:
        self.checks.write_text(
            "def bad(ctx, extra):\n    raise RuntimeError\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(pv.ConfigurationError, "exactly"):
            pv.resolve_custom_callable("validation/checks.py:bad", package_root=self.pkg)

    def test_bad_signature_kwargs(self) -> None:
        self.checks.write_text(
            "def bad(ctx, **kw):\n    raise RuntimeError\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(pv.ConfigurationError, "exactly"):
            pv.resolve_custom_callable("validation/checks.py:bad", package_root=self.pkg)

    def test_async_rejected(self) -> None:
        self.checks.write_text(
            "async def bad(ctx):\n    raise RuntimeError\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(pv.ConfigurationError, "async"):
            pv.resolve_custom_callable("validation/checks.py:bad", package_root=self.pkg)

    def test_valid_callable_loads_and_runs(self) -> None:
        ctx = pv.ValidationContext(
            descriptor=_descriptor(), base_revision="r", control_source=None,
            subject_source=None, package_root=self.pkg,
        )
        spec = pv.CheckSpec(
            check_id="act", capability="activation", validator="custom",
            required=False, config={"callable": "validation/checks.py:act_check"},
        )
        result = pv.evaluate_check(spec, ctx)
        self.assertEqual(result.status, pv.PASS)
        self.assertEqual(result.check_id, "act")

    def test_relabelled_result_rejected(self) -> None:
        self.checks.write_text(
            "from bigcherry import patch_validation as pv\n"
            "def sneaky(ctx):\n"
            "    return pv.ValidationResult(\n"
            "        check_id='other', capability='smoke', status='pass', summary='x')\n",
            encoding="utf-8",
        )
        ctx = pv.ValidationContext(
            descriptor=_descriptor(), base_revision="r", control_source=None,
            subject_source=None, package_root=self.pkg,
        )
        spec = pv.CheckSpec(
            check_id="act", capability="activation", validator="custom",
            required=True, config={"callable": "validation/checks.py:sneaky"},
        )
        result = pv.evaluate_check(spec, ctx)
        self.assertEqual(result.status, pv.ERROR)
        self.assertIn("different check", result.summary)

    def test_exception_becomes_error(self) -> None:
        self.checks.write_text(
            "from bigcherry import patch_validation as pv\n"
            "def boom(ctx):\n    raise RuntimeError('kapow')\n", encoding="utf-8"
        )
        ctx = pv.ValidationContext(
            descriptor=_descriptor(), base_revision="r", control_source=None,
            subject_source=None, package_root=self.pkg,
        )
        spec = pv.CheckSpec(
            check_id="act", capability="activation", validator="custom",
            required=True, config={"callable": "validation/checks.py:boom"},
        )
        result = pv.evaluate_check(spec, ctx)
        self.assertEqual(result.status, pv.ERROR)
        self.assertIn("kapow", result.summary)


def _plan_with(checks: list[tuple[str, bool, str]]) -> pv.ValidationPlan:
    """checks: (check_id, required, capability) triples."""
    return pv.ValidationPlan(
        patch_id="1201_test",
        checks=tuple(
            pv.CheckSpec(check_id=c, capability=cap, validator=cap, required=required)
            for c, required, cap in checks
        ),
        universal_capabilities=(),
        contract=None,
        required_capabilities=tuple(cap for c, required, cap in checks if required),
    )


def _result(check_id: str, status: str, capability: str = "apply") -> pv.ValidationResult:
    return pv.ValidationResult(
        check_id=check_id, capability=capability, status=status, summary=f"{status}"
    )


class VerdictTests(unittest.TestCase):
    def test_all_required_pass_is_eligible(self) -> None:
        plan = _plan_with([("a", True, "apply"), ("b", True, "build")])
        verdict = pv.compute_verdict(plan, {
            "a": _result("a", pv.PASS),
            "b": _result("b", pv.PASS, "build"),
        })
        self.assertTrue(verdict.eligible)
        self.assertEqual(verdict.reasons, ())
        self.assertFalse(verdict.blocked)
        self.assertEqual(verdict.errors, ())

    def test_required_check_blocked_is_not_eligible(self) -> None:
        plan = _plan_with([("a", True, "apply"), ("b", True, "build")])
        verdict = pv.compute_verdict(plan, {
            "a": _result("a", pv.BLOCKED),
            "b": _result("b", pv.PASS, "build"),
        })
        self.assertFalse(verdict.eligible)
        self.assertTrue(verdict.blocked)
        self.assertTrue(any("a" in r and "blocked" in r for r in verdict.reasons))

    def test_required_check_error_is_not_eligible(self) -> None:
        plan = _plan_with([("a", True, "apply"), ("b", True, "build")])
        verdict = pv.compute_verdict(plan, {
            "a": _result("a", pv.ERROR),
            "b": _result("b", pv.PASS, "build"),
        })
        self.assertFalse(verdict.eligible)
        self.assertIn("a", verdict.errors)

    def test_required_check_fail_is_not_eligible(self) -> None:
        plan = _plan_with([("a", True, "apply")])
        verdict = pv.compute_verdict(plan, {"a": _result("a", pv.FAIL)})
        self.assertFalse(verdict.eligible)

    def test_not_applicable_never_satisfies_required(self) -> None:
        plan = _plan_with([("a", True, "apply")])
        verdict = pv.compute_verdict(plan, {"a": _result("a", pv.NOT_APPLICABLE)})
        self.assertFalse(verdict.eligible)
        # demoted to error
        self.assertEqual(verdict.results[0].status, pv.ERROR)
        self.assertIn("a", verdict.errors)

    def test_not_applicable_ok_for_advisory(self) -> None:
        plan = _plan_with([("a", True, "apply"), ("adv", False, "smoke")])
        verdict = pv.compute_verdict(plan, {
            "a": _result("a", pv.PASS),
            "adv": _result("adv", pv.NOT_APPLICABLE, "smoke"),
        })
        self.assertTrue(verdict.eligible)

    def test_advisory_failure_does_not_block(self) -> None:
        plan = _plan_with([("a", True, "apply"), ("adv", False, "smoke")])
        verdict = pv.compute_verdict(plan, {
            "a": _result("a", pv.PASS),
            "adv": _result("adv", pv.FAIL, "smoke"),
        })
        self.assertTrue(verdict.eligible)

    def test_missing_result_is_structured_error(self) -> None:
        plan = _plan_with([("a", True, "apply"), ("b", True, "build")])
        verdict = pv.compute_verdict(plan, {"a": _result("a", pv.PASS)})
        self.assertFalse(verdict.eligible)
        self.assertIn("b", verdict.errors)


class RegistryAndVersionTests(unittest.TestCase):
    def test_framework_version_reexport(self) -> None:
        # B4: the constant is importable from patch_validation AND identical
        # to the registry-pinned value used in validation_digest.
        self.assertEqual(
            pv.VALIDATION_FRAMEWORK_VERSION, patch_registry.VALIDATION_FRAMEWORK_VERSION
        )
        self.assertEqual(pv.VALIDATION_FRAMEWORK_VERSION, "1")

    def test_unknown_validator_is_structured_error(self) -> None:
        spec = pv.CheckSpec(
            check_id="x", capability="apply", validator="fallback", required=True,
        )
        ctx = pv.ValidationContext(
            descriptor=_descriptor(), base_revision="r",
            control_source=None, subject_source=None,
        )
        result = pv.evaluate_check(spec, ctx)
        self.assertEqual(result.status, pv.ERROR)
        self.assertIn("fallback", result.summary)
        # no catch-all 'fallback' in the v1 set
        self.assertNotIn("fallback", pv.BUILTIN_VALIDATORS)

    def test_register_rejects_unknown_name(self) -> None:
        with self.assertRaises(pv.ConfigurationError):
            pv.register_builtin("totally-new", lambda spec, ctx: None)

    def test_validator_exception_becomes_error(self) -> None:
        pv.register_builtin("apply", lambda spec, ctx: (_ for _ in ()).throw(RuntimeError("x")))
        try:
            spec = pv.CheckSpec(
                check_id="x", capability="apply", validator="apply", required=True,
            )
            ctx = pv.ValidationContext(
                descriptor=_descriptor(), base_revision="r",
                control_source=None, subject_source=None,
            )
            result = pv.evaluate_check(spec, ctx)
            self.assertEqual(result.status, pv.ERROR)
        finally:
            pv.BUILTIN_REGISTRY.pop("apply", None)

    def test_plan_digest_stable_and_sensitive(self) -> None:
        specs = (
            pv.CheckSpec(check_id="a", capability="apply", validator="apply", required=True),
            pv.CheckSpec(check_id="b", capability="build", validator="build", required=True),
        )
        plan = pv.ValidationPlan(
            patch_id="1201", checks=specs, universal_capabilities=(), contract=None,
            required_capabilities=("apply", "build"),
        )
        first = pv.plan_digest(plan)
        second = pv.plan_digest(plan)
        self.assertEqual(first, second)
        changed = pv.ValidationPlan(
            patch_id="1201",
            checks=specs + (pv.CheckSpec(check_id="c", capability="smoke",
                                         validator="runtime-smoke", required=False),),
            universal_capabilities=(), contract=None,
            required_capabilities=("apply", "build"),
        )
        self.assertNotEqual(first, pv.plan_digest(changed))


if __name__ == "__main__":
    unittest.main()
