"""RD06 backend_reference orchestration tests -- hardware-free."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.patch import validation_campaign as vc  # noqa: E402


class _FakeBuildEvidence:
    effective_build_id = "same-build"
    effective_configure = {"CMAKE_BUILD_TYPE": "Release", "GGML_HIP": "ON"}
    verification = SimpleNamespace(to_dict=lambda: {})
    runtime_artifacts = {}

    def campaign_identity(self) -> dict[str, object]:
        return {"effective_build_id": self.effective_build_id}


def _fake_build_tree(*, name, hip_path, amdgpu_targets, workdir, targets, source, extra_cmake_args):
    return Path(f"/fake/{name}/bin")


def _fake_capture(
    build_dir, *, source_root, architecture, binary, requested_cmake_args, build_env, extra_binaries=(),
):
    return _FakeBuildEvidence()


def _fake_parity(control_evidence, subject_evidence, *, patch_id):
    return None


class _FakeSourceModule:
    REPO_ROOT = Path("R:/repo")

    def __init__(self) -> None:
        self.resolve_focals: list[str | None] = []

    def resolve_source_composition(self, source, *, focal=None, base_ref, base_repo):
        self.resolve_focals.append(focal)
        return base_ref, (() if focal is None else (focal,))

    def materialize_composition(
        self, *, base_repo, worktree_root, resolved_revision, composition, overlay_root, requested_revision,
    ):
        return Path(worktree_root) / "tree"

    def git_worktree_tree(self, source):
        return f"tree:{source}"


class RunRd06ContractCorrectnessTests(unittest.TestCase):
    def setUp(self) -> None:
        self._real_build_tree = vc.build_tree
        self._real_capture = vc.capture_completed_build_evidence
        self._real_parity = vc.assert_validation_subject_parity
        self._real_subprocess_run = vc.subprocess.run

        vc.build_tree = _fake_build_tree
        vc.capture_completed_build_evidence = _fake_capture
        vc.assert_validation_subject_parity = _fake_parity

    def tearDown(self) -> None:
        vc.build_tree = self._real_build_tree
        vc.capture_completed_build_evidence = self._real_capture
        vc.assert_validation_subject_parity = self._real_parity
        vc.subprocess.run = self._real_subprocess_run

    def _run(self, *, subject_ppl: float, control_ppl: float, uncertainty: float = 0.0275):
        run_dir = Path(tempfile.mkdtemp())
        source_module = _FakeSourceModule()

        def _fake_runner(argv, **kwargs):
            is_subject = "subject" in str(argv[0])
            ppl = subject_ppl if is_subject else control_ppl
            return SimpleNamespace(
                returncode=0,
                stdout=f"Final estimate: PPL = {ppl:.4f} +/- {uncertainty:.5f}",
                stderr="",
            )

        vc.subprocess.run = _fake_runner

        result = vc.run_rd06_contract_correctness(
            base_revision="a" * 40, hip_path=Path("H:/hip"), amdgpu_targets="gfx1201",
            worktree_root=Path("W:/worktrees"), build_root=Path("B:/build"),
            model=Path("M:/tierA-qwen4b-q6k.gguf"), corpus=Path("C:/wikitext2/test.txt"),
            run_dir=run_dir, _source_module=source_module,
        )
        return result, run_dir, source_module

    def test_within_tolerance_passes(self) -> None:
        result, run_dir, source_module = self._run(subject_ppl=10.3938, control_ppl=10.4463)

        correctness = result["results"]["backend_reference"]
        self.assertTrue(correctness.passed)

        artifact = run_dir / "artifacts" / "rd06-correctness-gfx1201.json"
        self.assertTrue(artifact.exists())

        doc = json.loads(artifact.read_text(encoding="utf-8"))
        self.assertEqual(doc["contract_id"], "RD06-RDNA4-WMMA-FA-CONFIG")
        self.assertEqual(doc["architecture"], "gfx1201")
        self.assertTrue(doc["comparison"]["ok"])

        self.assertEqual(
            source_module.resolve_focals,
            [None, "1203_rd050607_rdna4_wmma_fa_q6k_mmq"],
        )

    def test_large_ppl_divergence_is_a_correctness_failure(self) -> None:
        result, _, _ = self._run(subject_ppl=15.0, control_ppl=10.0, uncertainty=0.001)

        correctness = result["results"]["backend_reference"]
        self.assertFalse(correctness.passed)
        self.assertIn("sigma", correctness.detail)

    def test_wrong_architecture_fails_closed_before_materialization(self) -> None:
        source_module = _FakeSourceModule()

        with self.assertRaises(vc.PatchCampaignError):
            vc.run_rd06_contract_correctness(
                base_revision="a" * 40, hip_path=Path("H:/hip"), amdgpu_targets="gfx1100",
                worktree_root=Path("W:/worktrees"), build_root=Path("B:/build"),
                model=Path("M:/tierA-qwen4b-q6k.gguf"), corpus=Path("C:/wikitext2/test.txt"),
                run_dir=Path(tempfile.mkdtemp()), _source_module=source_module,
            )

        self.assertEqual(source_module.resolve_focals, [])

    def test_multi_architecture_build_fails_closed(self) -> None:
        source_module = _FakeSourceModule()

        with self.assertRaises(vc.PatchCampaignError):
            vc.run_rd06_contract_correctness(
                base_revision="a" * 40, hip_path=Path("H:/hip"), amdgpu_targets="gfx1201;gfx1100",
                worktree_root=Path("W:/worktrees"), build_root=Path("B:/build"),
                model=Path("M:/tierA-qwen4b-q6k.gguf"), corpus=Path("C:/wikitext2/test.txt"),
                run_dir=Path(tempfile.mkdtemp()), _source_module=source_module,
            )

        self.assertEqual(source_module.resolve_focals, [])


if __name__ == "__main__":
    unittest.main()
