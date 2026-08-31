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

    def require_rd08_correctness_evidence(self, *, subject_binary, control_binary):
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


def _fake_git_run(command, *args, **kwargs):
    class _Result:
        returncode = 0
        stdout = ""
        stderr = ""

    return _Result()


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

        def fake_probe(*, name, binary, model, hip_path, workdir, bench_prompt, bench_gen, disable_fusion):
            return logs[name]

        real_probe = vc._run_one_trace_probe
        vc._run_one_trace_probe = fake_probe
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
        from bigcherry.experiment import contract as ec

        proof = ec.evaluate_trigger_proof(result["evidence"])
        self.assertTrue(proof["passed"])

    def test_subject_no_hit_fails_trigger_proof(self) -> None:
        def fake_probe(*, name, binary, model, hip_path, workdir, bench_prompt, bench_gen, disable_fusion):
            return "nothing\n"

        real_probe = vc._run_one_trace_probe
        vc._run_one_trace_probe = fake_probe
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

        def fake_probe(*, name, binary, model, hip_path, workdir, bench_prompt, bench_gen, disable_fusion):
            return "BIGCHERRY_PATCH_HIT patch=x\n" if "subject" in name else "nothing\n"

        vc._run_one_trace_probe = fake_probe

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
            result.stdout = f"{metric} | {value} t/s\n"
            return result

        vc.subprocess.run = fake_run

    def tearDown(self) -> None:
        vc.build_tree = self._real_build_tree
        vc.capture_completed_build_evidence = self._real_capture
        vc._load_rd08_correctness_module = self._real_load_module
        vc._run_one_trace_probe = self._real_run_probe
        vc.subprocess.run = self._real_subprocess_run

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
        def fake_probe(*, name, binary, model, hip_path, workdir, bench_prompt, bench_gen, disable_fusion):
            return "nothing\n"  # neither subject nor control ever hits

        vc._run_one_trace_probe = fake_probe
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


if __name__ == "__main__":
    unittest.main()
