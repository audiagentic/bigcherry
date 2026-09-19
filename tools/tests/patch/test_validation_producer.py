"""PA36 step 0: hardware-free coverage for the generic validation-producer
protocol (tools/bigcherry/patch/validation_producer.py) -- exercised
against a synthetic producer, not a real migrated patch (none has
migrated yet)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.patch import validation as pv  # noqa: E402
from bigcherry.patch import validation_producer as vp  # noqa: E402


_PRODUCER_PY = '''
def run_correctness(ctx):
    from bigcherry.patch import validation_producer as vp
    return vp.ProducerResult(
        correctness={"disposition": "passed"},
        validation_build_identities={"control": {}, "subject": {}},
        activation_evidence=None,
        performance_evidence=None,
        trace_evidence=None,
        check_results=(),
        lane_effects=(),
        emitted_artifacts=frozenset({"correctness.json"}),
    )
'''

_PRODUCER_TOML = '''
schema = 1

[producer.correctness]
entrypoint = "producer.py"
callable = "run_correctness"
trace_probe = "skip"
standard_campaign = "skip"
correctness_evidence_cli = "forbid"
performance_benchmark_cli = "forbid"
artifacts = ["correctness.json"]

[producer.correctness.input.corpus]
type = "path"
required = true
'''


class _FakePatchDir:
    """Builds a real temp patches/<id>/validation/ tree on disk."""

    def __init__(self, tmp: Path, *, producer_toml: str = _PRODUCER_TOML, producer_py: str = _PRODUCER_PY) -> None:
        self.patch_dir = tmp / "0000_fake_patch"
        validation_dir = self.patch_dir / "validation"
        validation_dir.mkdir(parents=True)
        (validation_dir / "producer.toml").write_text(producer_toml, encoding="utf-8")
        (validation_dir / "producer.py").write_text(producer_py, encoding="utf-8")


class FatTargetPlanTests(unittest.TestCase):
    def test_cmake_value_is_semicolon_joined(self) -> None:
        plan = vp.FatTargetPlan(targets=("gfx1100", "gfx1201", "gfx1030"))
        self.assertEqual(plan.cmake_value, "gfx1100;gfx1201;gfx1030")

    def test_empty_targets_rejected(self) -> None:
        with self.assertRaises(vp.ValidationProducerError):
            vp.FatTargetPlan(targets=())

    def test_duplicate_targets_rejected(self) -> None:
        with self.assertRaises(vp.ValidationProducerError):
            vp.FatTargetPlan(targets=("gfx1100", "gfx1100"))


class ResolveProducerTests(unittest.TestCase):
    def test_resolves_real_producer_and_invokes_declared_callable(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            fake = _FakePatchDir(Path(td))
            selection = vp.resolve_producer(patch_dir=fake.patch_dir, producer_id="correctness")
            self.assertEqual(selection.spec.producer_id, "correctness")
            self.assertEqual(selection.spec.artifact_names, frozenset({"correctness.json"}))
            self.assertEqual(selection.spec.trace_probe, "skip")
            self.assertIn("corpus", selection.spec.inputs)
            self.assertTrue(selection.spec.inputs["corpus"].required)
            # The generic loader invokes EXACTLY the selected producer.
            result = selection.producer(None)
            self.assertEqual(result.emitted_artifacts, frozenset({"correctness.json"}))

    def test_missing_manifest_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            patch_dir = Path(td) / "0000_no_manifest"
            (patch_dir / "validation").mkdir(parents=True)
            with self.assertRaises(vp.ValidationProducerError):
                vp.resolve_producer(patch_dir=patch_dir, producer_id="correctness")

    def test_unknown_producer_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            fake = _FakePatchDir(Path(td))
            with self.assertRaises(vp.ValidationProducerError):
                vp.resolve_producer(patch_dir=fake.patch_dir, producer_id="nonexistent")

    def test_absolute_entrypoint_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            toml = _PRODUCER_TOML.replace('entrypoint = "producer.py"', 'entrypoint = "/etc/passwd"')
            fake = _FakePatchDir(Path(td), producer_toml=toml)
            with self.assertRaises(vp.ValidationProducerError):
                vp.resolve_producer(patch_dir=fake.patch_dir, producer_id="correctness")

    def test_traversal_entrypoint_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            toml = _PRODUCER_TOML.replace(
                'entrypoint = "producer.py"', 'entrypoint = "../../secrets.py"',
            )
            fake = _FakePatchDir(Path(td), producer_toml=toml)
            with self.assertRaises(vp.ValidationProducerError):
                vp.resolve_producer(patch_dir=fake.patch_dir, producer_id="correctness")

    def test_missing_callable_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            toml = _PRODUCER_TOML.replace('callable = "run_correctness"', 'callable = "does_not_exist"')
            fake = _FakePatchDir(Path(td), producer_toml=toml)
            with self.assertRaises(vp.ValidationProducerError):
                vp.resolve_producer(patch_dir=fake.patch_dir, producer_id="correctness")

    def test_duplicate_artifact_names_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            toml = _PRODUCER_TOML.replace(
                'artifacts = ["correctness.json"]',
                'artifacts = ["correctness.json", "correctness.json"]',
            )
            fake = _FakePatchDir(Path(td), producer_toml=toml)
            with self.assertRaises(vp.ValidationProducerError):
                vp.resolve_producer(patch_dir=fake.patch_dir, producer_id="correctness")

    def test_invalid_artifact_name_with_path_separator_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            toml = _PRODUCER_TOML.replace(
                'artifacts = ["correctness.json"]',
                'artifacts = ["../escape.json"]',
            )
            fake = _FakePatchDir(Path(td), producer_toml=toml)
            with self.assertRaises(vp.ValidationProducerError):
                vp.resolve_producer(patch_dir=fake.patch_dir, producer_id="correctness")

    def test_unknown_policy_enum_value_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            toml = _PRODUCER_TOML.replace('trace_probe = "skip"', 'trace_probe = "maybe"')
            fake = _FakePatchDir(Path(td), producer_toml=toml)
            with self.assertRaises(vp.ValidationProducerError):
                vp.resolve_producer(patch_dir=fake.patch_dir, producer_id="correctness")


class ProducerInputValidationTests(unittest.TestCase):
    def _spec(self) -> vp.ProducerSpec:
        return vp.ProducerSpec(
            patch_id="0000_fake_patch", producer_id="correctness",
            entrypoint=Path("producer.py"), callable_name="run_correctness",
            trace_probe="skip", standard_campaign="skip",
            correctness_evidence_cli="forbid", performance_benchmark_cli="forbid",
            artifact_names=frozenset({"correctness.json"}),
            inputs={"corpus": vp.ProducerInputSpec(name="corpus", type="path", required=True)},
        )

    def test_undeclared_input_rejected(self) -> None:
        with self.assertRaises(vp.ValidationProducerError):
            vp.validate_producer_inputs(self._spec(), {"unknown": "value"})

    def test_missing_required_input_rejected(self) -> None:
        with self.assertRaises(vp.ValidationProducerError):
            vp.validate_producer_inputs(self._spec(), {})

    def test_valid_input_accepted(self) -> None:
        result = vp.validate_producer_inputs(self._spec(), {"corpus": "/tmp/x.txt"})
        self.assertEqual(result, {"corpus": "/tmp/x.txt"})


class CliCompatibilityTests(unittest.TestCase):
    def _spec(self, **overrides) -> vp.ProducerSpec:
        defaults = dict(
            patch_id="0000_fake_patch", producer_id="correctness",
            entrypoint=Path("producer.py"), callable_name="run_correctness",
            trace_probe="skip", standard_campaign="skip",
            correctness_evidence_cli="forbid", performance_benchmark_cli="forbid",
            artifact_names=frozenset(), inputs={},
        )
        defaults.update(overrides)
        return vp.ProducerSpec(**defaults)

    def test_correctness_evidence_ambiguity_rejected_when_forbidden(self) -> None:
        with self.assertRaises(vp.ValidationProducerError):
            vp.validate_producer_cli_compatibility(
                self._spec(), correctness_evidence_requested=True,
                performance_benchmark_requested=False,
            )

    def test_correctness_evidence_allowed_when_policy_says_allow(self) -> None:
        vp.validate_producer_cli_compatibility(
            self._spec(correctness_evidence_cli="allow"),
            correctness_evidence_requested=True, performance_benchmark_requested=False,
        )  # must not raise

    def test_performance_benchmark_rejected_when_forbidden(self) -> None:
        with self.assertRaises(vp.ValidationProducerError):
            vp.validate_producer_cli_compatibility(
                self._spec(), correctness_evidence_requested=False,
                performance_benchmark_requested=True,
            )

    def test_no_conflict_when_neither_requested(self) -> None:
        vp.validate_producer_cli_compatibility(
            self._spec(), correctness_evidence_requested=False,
            performance_benchmark_requested=False,
        )  # must not raise


def _empty_plan_and_context() -> tuple[pv.ValidationPlan, pv.ValidationContext]:
    check = pv.CheckSpec(
        check_id="correctness-check", capability="correctness", validator="custom",
        required=True, config={"callable": "checks.py:check"},
    )
    plan = pv.ValidationPlan(
        patch_id="p", checks=(check,), universal_capabilities=(),
    )
    context = pv.ValidationContext(
        descriptor=None, base_revision="a" * 40, control_source=None, subject_source=None,
    )
    return plan, context


class ProducerResultTests(unittest.TestCase):
    def _make_result(self, **overrides) -> vp.ProducerResult:
        defaults = dict(
            correctness=None,
            validation_build_identities={"control": {}, "subject": {}},
            activation_evidence=None, performance_evidence=None, trace_evidence=None,
            check_results=(), lane_effects=(), emitted_artifacts=frozenset(),
        )
        defaults.update(overrides)
        return vp.ProducerResult(**defaults)

    def test_validation_build_identities_must_be_exactly_control_subject(self) -> None:
        with self.assertRaises(vp.ValidationProducerError):
            self._make_result(validation_build_identities={"control": {}})
        with self.assertRaises(vp.ValidationProducerError):
            self._make_result(validation_build_identities={"control": {}, "subject": {}, "extra": {}})
        with self.assertRaises(vp.ValidationProducerError):
            self._make_result(validation_build_identities={"tune": {}, "replay": {}})

    def test_valid_result_constructs(self) -> None:
        result = self._make_result()
        self.assertEqual(set(result.validation_build_identities), {"control", "subject"})

    def test_undeclared_emitted_artifact_rejected(self) -> None:
        spec = vp.ProducerSpec(
            patch_id="p", producer_id="prod", entrypoint=Path("producer.py"),
            callable_name="run", trace_probe="skip", standard_campaign="skip",
            correctness_evidence_cli="forbid", performance_benchmark_cli="forbid",
            artifact_names=frozenset({"correctness.json"}), inputs={},
        )
        result = self._make_result(emitted_artifacts=frozenset({"correctness.json", "sneaky.json"}))
        plan, context = _empty_plan_and_context()
        with self.assertRaises(vp.ValidationProducerError):
            vp.validate_producer_result(spec, result, plan=plan, context=context)

    def test_declared_artifact_accepted(self) -> None:
        spec = vp.ProducerSpec(
            patch_id="p", producer_id="prod", entrypoint=Path("producer.py"),
            callable_name="run", trace_probe="skip", standard_campaign="skip",
            correctness_evidence_cli="forbid", performance_benchmark_cli="forbid",
            artifact_names=frozenset({"correctness.json"}), inputs={},
        )
        result = self._make_result(emitted_artifacts=frozenset({"correctness.json"}))
        plan, context = _empty_plan_and_context()
        vp.validate_producer_result(spec, result, plan=plan, context=context)  # must not raise

    def test_check_result_scoped_to_wrong_contract_rejected(self) -> None:
        spec = vp.ProducerSpec(
            patch_id="p", producer_id="prod", entrypoint=Path("producer.py"),
            callable_name="run", trace_probe="skip", standard_campaign="skip",
            correctness_evidence_cli="forbid", performance_benchmark_cli="forbid",
            artifact_names=frozenset(), inputs={},
        )
        plan, context = _empty_plan_and_context()
        bad = vp.ProducerCheckResult(
            check_id="correctness-check",
            contract_ids=("some-other-contract",),
            validation_result=pv.ValidationResult(
                check_id="correctness-check", capability="correctness",
                status=pv.PASS, summary="ok",
            ),
        )
        result = self._make_result(check_results=(bad,))
        with self.assertRaises(vp.ValidationProducerError):
            vp.validate_producer_result(spec, result, plan=plan, context=context)

    def test_valid_typed_check_result_accepted(self) -> None:
        spec = vp.ProducerSpec(
            patch_id="p", producer_id="prod", entrypoint=Path("producer.py"),
            callable_name="run", trace_probe="skip", standard_campaign="skip",
            correctness_evidence_cli="forbid", performance_benchmark_cli="forbid",
            artifact_names=frozenset(), inputs={},
        )
        plan, context = _empty_plan_and_context()
        good = vp.ProducerCheckResult(
            check_id="correctness-check",
            contract_ids=(),
            validation_result=pv.ValidationResult(
                check_id="correctness-check", capability="correctness",
                status=pv.PASS, summary="ok",
            ),
        )
        result = self._make_result(check_results=(good,))
        vp.validate_producer_result(spec, result, plan=plan, context=context)  # must not raise


if __name__ == "__main__":
    unittest.main()
