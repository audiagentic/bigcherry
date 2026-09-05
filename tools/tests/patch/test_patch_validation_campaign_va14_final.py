"""VA14 final slice tests: RD08's real full-qualification orchestration
(run_rd08_contract_correctness, run_rd08_contract_trigger,
run_rd08_contract_qualification) -- hardware-free via injected fakes for
the RD08 correctness producer module, build_tree(),
capture_completed_build_evidence(), subprocess.run(), and
_run_one_trace_probe(), consistent with VA14's established pattern.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.experiment import contract as ec  # noqa: E402
from bigcherry.patch import validation_campaign as vc  # noqa: E402

_CONTRACTS = ec.load_contracts(
    Path(__file__).resolve().parents[3] / "config" / "experiment-contracts.toml"
)
RD08 = _CONTRACTS.contracts["RD08-Q6K-MMVQ-VDR2"]


class _FakeRow:
    def __init__(self, shape_name: str, seed: int, ok: bool) -> None:
        self.shape_name = shape_name
        self.seed = seed
        self.ok = ok


class _Rd08CorrectnessError(RuntimeError):
    pass


class _FakeRd08CorrectnessModule:
    Rd08CorrectnessError = _Rd08CorrectnessError

    def __init__(self, *, fail: bool = False) -> None:
        self._fail = fail
        self.rows = tuple(_FakeRow(f"shape{i}", seed, True) for i in range(2) for seed in (1, 2))

    def materialize_rd08_variants(self, *, base_repo, worktree_root, base_revision):
        return Path("subject_src"), Path("control_src")

    def require_rd08_correctness_evidence(self, *, subject_binary, control_binary, runner=None):
        if self._fail:
            raise self.Rd08CorrectnessError("mismatch at shape=shape0 seed=1")
        return self.rows


class _FakeBuildEvidence:
    def __init__(self, tag: str) -> None:
        self.effective_build_id = f"build-{tag}"
        self.effective_configure = {"CMAKE_BUILD_TYPE": "Release"}
        self.verification = SimpleNamespace(to_dict=lambda: {})
        self.runtime_artifacts = {}

    def campaign_identity(self) -> dict:
        return {"effective_build_id": self.effective_build_id}


def _fake_build_tree(*, name, hip_path, amdgpu_targets, workdir, targets, source, extra_cmake_args):
    return Path(f"/fake/{name}/bin")


def _fake_capture(build_dir, *, source_root, architecture, binary, requested_cmake_args, build_env, extra_binaries=()):
    return _FakeBuildEvidence(str(binary).replace("/", "_"))


def _make_fake_probe(content_for_name):
    """content_for_name(name) -> str. Also writes the log file to disk --
    run_rd08_contract_trigger() now hashes the real file the same way
    _run_one_trace_probe() does, so a fake that only returns a string
    (without writing it) leaves nothing to hash."""
    def fake_probe(*, name, binary, model, hip_path, workdir, bench_prompt, bench_gen, disable_fusion):
        content = content_for_name(name)
        log_path = Path(workdir) / "logs" / f"activation-{name}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(content, encoding="utf-8")
        return content
    return fake_probe


def _fake_git_run(command, *args, **kwargs):
    class _Result:
        returncode = 0
        stdout = ""
        stderr = ""

    return _Result()


class LoadRd08CorrectnessModuleTests(unittest.TestCase):
    def test_loads_the_real_module_without_crashing_on_dataclass_decoration(self) -> None:
        # VA15 real-hardware finding (req_bc329f6ae30c4e4c follow-up):
        # importlib.util.module_from_spec() does not register the module in
        # sys.modules -- @dataclass (Rd08Shape, ShapeSeedComparison inside
        # rd08_correctness.py) resolves its owning module via
        # sys.modules[cls.__module__] during decoration and crashed with
        # AttributeError: 'NoneType' object has no attribute '__dict__'
        # before sys.modules[spec.name] = module was added.
        module = vc._load_rd08_correctness_module()
        self.assertTrue(hasattr(module, "RD08_SHAPES"))
        self.assertEqual(len(module.RD08_SHAPES), 5)
        self.assertTrue(hasattr(module, "materialize_rd08_variants"))
        self.assertTrue(hasattr(module, "require_rd08_correctness_evidence"))


class RunRd08ContractCorrectnessTests(unittest.TestCase):
    def setUp(self) -> None:
        self._real_build_tree = vc.build_tree
        self._real_capture = vc.capture_completed_build_evidence
        self._real_subprocess_run = vc.subprocess.run
        vc.build_tree = _fake_build_tree
        vc.capture_completed_build_evidence = _fake_capture
        vc.subprocess.run = _fake_git_run  # psi.git_worktree_tree() shells out to git

    def tearDown(self) -> None:
        vc.build_tree = self._real_build_tree
        vc.capture_completed_build_evidence = self._real_capture
        vc.subprocess.run = self._real_subprocess_run

    def test_success_produces_a_passing_bit_identical_result(self) -> None:
        run_dir = Path(tempfile.mkdtemp())
        result = vc.run_rd08_contract_correctness(
            base_revision="a" * 40, hip_path=Path("H:/hip"), amdgpu_targets="gfx1100",
            worktree_root=Path("W:/worktrees"), build_root=Path("B:/build"),
            build_env={}, run_dir=run_dir, _module=_FakeRd08CorrectnessModule(),
        )
        self.assertTrue(result["results"]["bit_identical"].passed)
        self.assertTrue((run_dir / "artifacts" / "rd08-correctness.json").exists())

    def test_correctness_failure_produces_a_failing_result_not_a_raise(self) -> None:
        run_dir = Path(tempfile.mkdtemp())
        result = vc.run_rd08_contract_correctness(
            base_revision="a" * 40, hip_path=Path("H:/hip"), amdgpu_targets="gfx1100",
            worktree_root=Path("W:/worktrees"), build_root=Path("B:/build"),
            build_env={}, run_dir=run_dir, _module=_FakeRd08CorrectnessModule(fail=True),
        )
        self.assertFalse(result["results"]["bit_identical"].passed)
        self.assertIn("mismatch", result["results"]["bit_identical"].detail)


class RunRd08ContractTriggerTests(unittest.TestCase):
    def test_subject_hit_control_miss_produces_valid_trigger_proof(self) -> None:
        logs = {"rd08-trigger-subject": "BIGCHERRY_PATCH_HIT patch=x\n", "rd08-trigger-control": "nothing\n"}

        real_probe = vc._run_one_trace_probe
        vc._run_one_trace_probe = _make_fake_probe(lambda name: logs[name])
        try:
            run_dir = Path(tempfile.mkdtemp())
            result = vc.run_rd08_contract_trigger(
                marker_regex="BIGCHERRY_PATCH_HIT", control_binary=Path("control_bin"),
                subject_binary=Path("subject_bin"), model=Path("m.gguf"), hip_path=Path("H:/hip"),
                workdir=run_dir, run_dir=run_dir,
            )
        finally:
            vc._run_one_trace_probe = real_probe

        self.assertTrue(result["subject_hit"])
        self.assertFalse(result["control_hit"])
        actual_bytes = (run_dir / result["subject_log_path"]).read_bytes()
        self.assertEqual(
            result["subject_log_artifact"]["sha256"],
            __import__("hashlib").sha256(actual_bytes).hexdigest(),
        )
        from bigcherry.experiment import contract as ec

        proof = ec.evaluate_trigger_proof(result["evidence"])
        self.assertTrue(proof["passed"])

    def test_subject_no_hit_fails_trigger_proof(self) -> None:
        real_probe = vc._run_one_trace_probe
        vc._run_one_trace_probe = _make_fake_probe(lambda name: "nothing\n")
        try:
            run_dir = Path(tempfile.mkdtemp())
            result = vc.run_rd08_contract_trigger(
                marker_regex="BIGCHERRY_PATCH_HIT", control_binary=Path("control_bin"),
                subject_binary=Path("subject_bin"), model=Path("m.gguf"), hip_path=Path("H:/hip"),
                workdir=run_dir, run_dir=run_dir,
            )
        finally:
            vc._run_one_trace_probe = real_probe

        self.assertFalse(result["subject_hit"])
        from bigcherry.experiment import contract as ec

        proof = ec.evaluate_trigger_proof(result["evidence"])
        self.assertFalse(proof["passed"])


class RunRd08ContractQualificationTests(unittest.TestCase):
    """Full composition, end to end, entirely hardware-free."""

    def setUp(self) -> None:
        self._real_build_tree = vc.build_tree
        self._real_capture = vc.capture_completed_build_evidence
        self._real_load_module = vc._load_rd08_correctness_module
        self._real_run_probe = vc._run_one_trace_probe
        self._real_subprocess_run = vc.subprocess.run
        vc.build_tree = _fake_build_tree
        vc.capture_completed_build_evidence = _fake_capture
        vc._load_rd08_correctness_module = lambda: _FakeRd08CorrectnessModule()

        vc._run_one_trace_probe = _make_fake_probe(
            lambda name: "BIGCHERRY_PATCH_HIT patch=x\n" if "subject" in name else "nothing\n"
        )

        def fake_run(command, *args, **kwargs):
            class _Result:
                returncode = 0
                stdout = ""
                stderr = ""

            if command and command[0] == "git":
                return _Result()
            binary = command[0]
            metric = "tg128" if "-n" in command and command[command.index("-n") + 1] == "128" else "pp512"
            value = {"control_bin": 100.0, "subject_bin": 105.0}[binary]
            result = _Result()
            result.stdout = f"ggml_cuda_init: found 1 ROCm devices\n{metric} | {value} t/s\n"
            return result

        vc.subprocess.run = fake_run

    def tearDown(self) -> None:
        vc.build_tree = self._real_build_tree
        vc.capture_completed_build_evidence = self._real_capture
        vc._load_rd08_correctness_module = self._real_load_module
        vc._run_one_trace_probe = self._real_run_probe
        vc.subprocess.run = self._real_subprocess_run

    def test_control_hit_invalidates_promotion_even_with_a_positive_subject_hit(self) -> None:
        # GPT round 4 (req_4544a9240b6d45df): evaluate_trigger_proof() only
        # checks positive-role lanes -- a control (unpatched) binary that
        # ALSO shows the marker means the negative control itself is
        # invalid, and this must be enforced explicitly, not left to
        # evaluate_trigger_proof()'s positive-only scope.
        vc._run_one_trace_probe = _make_fake_probe(
            lambda name: "BIGCHERRY_PATCH_HIT patch=x\n"  # both subject AND control hit
        )
        run_dir = Path(tempfile.mkdtemp())
        result = vc.run_rd08_contract_qualification(
            contract=RD08, descriptor=SimpleNamespace(), base_revision="a" * 40,
            control_binary=Path("control_bin"), subject_binary=Path("subject_bin"),
            model=Path("m.gguf"), model_ref=RD08.positive.models[0],
            marker_regex="BIGCHERRY_PATCH_HIT", hip_path=Path("H:/hip"),
            amdgpu_targets="gfx1100", worktree_root=Path("W:/worktrees"),
            build_root=Path("B:/build"), build_env={}, run_dir=run_dir,
            control_build_identity={"effective_build_id": "c1"},
            subject_build_identity={"effective_build_id": "s1"},
            pairs=2,
        )
        self.assertTrue(result["trigger"]["control_hit"])
        self.assertFalse(result["trigger_proof"]["passed"])
        self.assertFalse(result["promotion"]["passed"])
        self.assertEqual(result["promotion"]["status"], "invalid")

    def test_all_gates_passing_composes_into_a_passing_promotion(self) -> None:
        run_dir = Path(tempfile.mkdtemp())
        result = vc.run_rd08_contract_qualification(
            contract=RD08, descriptor=SimpleNamespace(), base_revision="a" * 40,
            control_binary=Path("control_bin"), subject_binary=Path("subject_bin"),
            model=Path("m.gguf"), model_ref=RD08.positive.models[0],
            marker_regex="BIGCHERRY_PATCH_HIT", hip_path=Path("H:/hip"),
            amdgpu_targets="gfx1100", worktree_root=Path("W:/worktrees"),
            build_root=Path("B:/build"), build_env={}, run_dir=run_dir,
            control_build_identity={"effective_build_id": "c1"},
            subject_build_identity={"effective_build_id": "s1"},
            pairs=2,
        )
        self.assertTrue(result["promotion"]["passed"], result["promotion"])
        self.assertTrue((run_dir / "artifacts" / "contract-qualification.json").exists())

    def test_failing_trigger_makes_promotion_invalid_not_silently_failed(self) -> None:
        vc._run_one_trace_probe = _make_fake_probe(lambda name: "nothing\n")  # neither ever hits
        run_dir = Path(tempfile.mkdtemp())
        result = vc.run_rd08_contract_qualification(
            contract=RD08, descriptor=SimpleNamespace(), base_revision="a" * 40,
            control_binary=Path("control_bin"), subject_binary=Path("subject_bin"),
            model=Path("m.gguf"), model_ref=RD08.positive.models[0],
            marker_regex="BIGCHERRY_PATCH_HIT", hip_path=Path("H:/hip"),
            amdgpu_targets="gfx1100", worktree_root=Path("W:/worktrees"),
            build_root=Path("B:/build"), build_env={}, run_dir=run_dir,
            control_build_identity={"effective_build_id": "c1"},
            subject_build_identity={"effective_build_id": "s1"},
            pairs=2,
        )
        self.assertFalse(result["promotion"]["passed"])
        self.assertEqual(result["promotion"]["status"], "invalid")

    def test_failing_correctness_makes_promotion_fail(self) -> None:
        vc._load_rd08_correctness_module = lambda: _FakeRd08CorrectnessModule(fail=True)
        run_dir = Path(tempfile.mkdtemp())
        result = vc.run_rd08_contract_qualification(
            contract=RD08, descriptor=SimpleNamespace(), base_revision="a" * 40,
            control_binary=Path("control_bin"), subject_binary=Path("subject_bin"),
            model=Path("m.gguf"), model_ref=RD08.positive.models[0],
            marker_regex="BIGCHERRY_PATCH_HIT", hip_path=Path("H:/hip"),
            amdgpu_targets="gfx1100", worktree_root=Path("W:/worktrees"),
            build_root=Path("B:/build"), build_env={}, run_dir=run_dir,
            control_build_identity={"effective_build_id": "c1"},
            subject_build_identity={"effective_build_id": "s1"},
            pairs=2,
        )
        self.assertFalse(result["promotion"]["passed"])
        self.assertNotEqual(result["promotion"]["status"], "invalid")


class EndToEndEligibilityCompositionTests(unittest.TestCase):
    """GPT round 4 (req_4544a9240b6d45df): proves the exact seam that was
    broken -- RD08's real validation.toml checks that read
    ctx.performance_evidence (correctness/performance/controls, all
    validator="autotune-campaign") and ctx.trace_evidence (activation,
    validator="trace-marker") now reach real PASS with the evidence shapes
    --run-rd08-contract actually constructs in run(), and that combining a
    passing adapter verdict with a passing contract_promotions entry
    produces eligible_for_validated_state=True end to end. Does not invoke
    the full CLI run() (source materialization/build/cmake are out of
    scope for a hardware-free test) -- exercises the real check-evaluation
    and eligibility-composition functions directly, using RD08's own
    validation.toml check shapes read from the real committed file."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.run_dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _bind(self, relative_path: str, content: str) -> dict:
        import hashlib

        target = self.run_dir / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return {"path": relative_path, "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}

    def test_rd08_autotune_campaign_and_trace_marker_checks_pass_with_real_evidence_shapes(self) -> None:
        from bigcherry.patch import validation as pv
        import json

        performance_evidence = {
            "artifact": self._bind(
                "performance.json",
                json.dumps({"campaign_id": "digest123", "passed": True, "target_kernel_gain_pct": 1.5}),
            )
        }
        trace_evidence = {
            "positive": {
                "marker_regex": "BIGCHERRY_PATCH_HIT",
                "artifact": self._bind("logs/subject.log", "BIGCHERRY_PATCH_HIT patch=x\n"),
            },
            "negative": {
                "marker_regex": "BIGCHERRY_PATCH_HIT",
                "artifact": self._bind("logs/control.log", "nothing\n"),
            },
        }
        ctx = pv.ValidationContext(
            descriptor=None, base_revision="a" * 40, control_source=None, subject_source=None,
            run_dir=self.run_dir, performance_evidence=performance_evidence,
            trace_evidence=trace_evidence,
        )
        # Real check shapes from patches/1204_rd08_q6k_mmvq_vdr2/validation.toml.
        for check_id in ("correctness", "performance", "controls"):
            spec = pv.CheckSpec(check_id, check_id, "autotune-campaign", True, {})
            result = pv.evaluate_check(spec, ctx)
            self.assertEqual(result.status, pv.PASS, f"{check_id}: {result.summary}")

        activation_spec = pv.CheckSpec(
            "activation", "activation", "trace-marker", True,
            {"marker-regex": "BIGCHERRY_PATCH_HIT"},
        )
        activation_result = pv.evaluate_check(activation_spec, ctx)
        self.assertEqual(activation_result.status, pv.PASS, activation_result.summary)

    def test_activation_evidence_executed_only_when_subject_hit_and_not_control_hit(self) -> None:
        from bigcherry.patch.activation import ActivationEvidence, verdict

        both_hit = ActivationEvidence(status="not_executed", mechanism="rd08-trigger-marker", detail="")
        self.assertEqual(verdict(both_hit, correctness_passed=None), "failed-activation")

        subject_only = ActivationEvidence(status="executed", mechanism="rd08-trigger-marker", detail="")
        self.assertEqual(verdict(subject_only, correctness_passed=None), "activation-verified")

    def test_full_composition_adapter_pass_plus_contract_pass_is_eligible(self) -> None:
        from dataclasses import dataclass

        @dataclass
        class _Verdict:
            eligible: bool
            reasons: tuple = ()

        @dataclass
        class _Descriptor:
            experiment_contracts: tuple

        descriptor = _Descriptor(experiment_contracts=("RD08-Q6K-MMVQ-VDR2",))
        adapter_verdict = _Verdict(eligible=True)
        contract_promotions = {"RD08-Q6K-MMVQ-VDR2": {"passed": True}}
        self.assertTrue(
            vc.compute_persisted_validation_eligible(
                descriptor, adapter_verdict, contract_promotions,
                activation_disposition="activation-verified",
                correctness={"disposition": "passed"},
            )
        )
        # And the failure side: adapter PASS alone (no real promotion) must
        # NOT be eligible -- the exact bug this whole slice exists to fix.
        self.assertFalse(
            vc.compute_persisted_validation_eligible(
                descriptor, adapter_verdict, {},
                activation_disposition="activation-verified",
                correctness={"disposition": "passed"},
            )
        )


class ArtifactBindingTests(unittest.TestCase):
    """GPT round 5 (req_12dd706a42e341bd): performance.json and the two
    real RD08 trigger logs must be tracked by _artifact_refs() (used by
    evidence.py::make_record() to populate campaign_artifacts) -- writing
    them alone is not sufficient if the tracker never looks for them."""

    def test_performance_json_and_trigger_logs_are_bound_into_artifact_refs(self) -> None:
        import hashlib
        from bigcherry.patch import evidence as patch_evidence

        run_dir = Path(tempfile.mkdtemp())
        perf_path = run_dir / "performance.json"
        perf_path.write_text('{"campaign_id": "x", "passed": true}', encoding="utf-8")
        (run_dir / "logs").mkdir(parents=True, exist_ok=True)
        (run_dir / "logs" / "activation-rd08-trigger-subject.log").write_text("hit\n", encoding="utf-8")
        (run_dir / "logs" / "activation-rd08-trigger-control.log").write_text("miss\n", encoding="utf-8")

        refs = {r["path"]: r["sha256"] for r in patch_evidence._artifact_refs(run_dir)}
        self.assertIn("performance.json", refs)
        self.assertEqual(refs["performance.json"], hashlib.sha256(perf_path.read_bytes()).hexdigest())
        self.assertIn("logs/activation-rd08-trigger-subject.log", refs)
        self.assertIn("logs/activation-rd08-trigger-control.log", refs)

    def test_rd08_trigger_json_carries_real_hash_refs_to_both_logs(self) -> None:
        real_probe = vc._run_one_trace_probe
        vc._run_one_trace_probe = _make_fake_probe(
            lambda name: "BIGCHERRY_PATCH_HIT patch=x\n" if "subject" in name else "nothing\n"
        )
        run_dir = Path(tempfile.mkdtemp())
        try:
            result = vc.run_rd08_contract_trigger(
                marker_regex="BIGCHERRY_PATCH_HIT", control_binary=Path("control_bin"),
                subject_binary=Path("subject_bin"), model=Path("m.gguf"), hip_path=Path("H:/hip"),
                workdir=run_dir, run_dir=run_dir,
            )
        finally:
            vc._run_one_trace_probe = real_probe

        import json

        doc = json.loads((run_dir / "artifacts" / "rd08-trigger.json").read_text(encoding="utf-8"))
        self.assertIn("artifact", doc["positive"])
        self.assertIn("artifact", doc["control"])
        self.assertEqual(doc["positive"]["artifact"], result["subject_log_artifact"])
        self.assertEqual(doc["control"]["artifact"], result["control_log_artifact"])


if __name__ == "__main__":
    unittest.main()
