"""RD04 backend_reference/ppl_equality orchestration tests -- hardware-free."""

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
        self, *, base_repo, worktree_root, resolved_revision, composition,
        overlay_root, requested_revision,
    ):
        return Path(worktree_root) / "tree"

    def git_worktree_tree(self, source):
        return f"tree:{source}"


class RunRd04ContractCorrectnessTests(unittest.TestCase):
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

    def _run(
        self, *, architecture: str, subject_ppl: float, control_ppl: float,
        uncertainty: float = 0.0275,
    ):
        run_dir = Path(tempfile.mkdtemp())
        source_module = _FakeSourceModule()
        calls: list[list[str]] = []

        def _fake_runner(argv, **kwargs):
            calls.append(list(argv))
            is_subject = "subject" in str(argv[0])
            ppl = subject_ppl if is_subject else control_ppl
            return SimpleNamespace(
                returncode=0,
                stdout=f"Final estimate: PPL = {ppl:.4f} +/- {uncertainty:.5f}",
                stderr="",
            )

        vc.subprocess.run = _fake_runner

        result = vc.run_rd04_contract_correctness(
            base_revision="a" * 40,
            hip_path=Path("H:/hip"),
            amdgpu_targets=architecture,
            worktree_root=Path("W:/worktrees"),
            build_root=Path("B:/build"),
            model=Path("M:/tierA-qwen4b-q6k.gguf"),
            corpus=Path("C:/wikitext2/test.txt"),
            run_dir=run_dir,
            _source_module=source_module,
        )
        return result, run_dir, source_module, calls

    def test_each_contract_architecture_can_pass(self) -> None:
        for architecture in ("gfx1100", "gfx1201", "gfx1030"):
            with self.subTest(architecture=architecture):
                result, run_dir, source_module, calls = self._run(
                    architecture=architecture,
                    subject_ppl=10.3938,
                    control_ppl=10.4463,
                )

                backend_reference = result["results"]["backend_reference"]
                ppl_equality = result["results"]["ppl_equality"]

                self.assertTrue(backend_reference.passed)
                self.assertTrue(ppl_equality.passed)
                self.assertEqual(backend_reference.check, "backend_reference")
                self.assertEqual(ppl_equality.check, "ppl_equality")
                self.assertEqual(backend_reference.detail, ppl_equality.detail)

                # One subject/control PPL pair feeds both CorrectnessResults.
                self.assertEqual(len(calls), 2)
                for argv in calls:
                    self.assertEqual(argv[argv.index("-fa") + 1], "on")
                    self.assertEqual(argv[argv.index("-ctk") + 1], "bf16")
                    self.assertEqual(argv[argv.index("-ctv") + 1], "bf16")

                artifact = run_dir / "artifacts" / f"rd04-correctness-{architecture}.json"
                self.assertTrue(artifact.exists())

                doc = json.loads(artifact.read_text(encoding="utf-8"))
                self.assertEqual(doc["contract_id"], "RD04-BF16-FLASH-ATTN-TILE")
                self.assertEqual(doc["architecture"], architecture)
                self.assertTrue(doc["comparison"]["ok"])
                self.assertTrue(doc["results"]["backend_reference"]["passed"])
                self.assertTrue(doc["results"]["ppl_equality"]["passed"])
                self.assertEqual(
                    doc["perplexity_extra_args"],
                    ["-fa", "on", "-ctk", "bf16", "-ctv", "bf16"],
                )
                self.assertEqual(
                    source_module.resolve_focals,
                    [None, "1202_rd04_bf16_flash_attn_tile"],
                )

    def test_large_ppl_divergence_fails_both_checks(self) -> None:
        result, _, _, calls = self._run(
            architecture="gfx1100",
            subject_ppl=15.0,
            control_ppl=10.0,
            uncertainty=0.001,
        )

        self.assertEqual(len(calls), 2)
        for check in ("backend_reference", "ppl_equality"):
            correctness = result["results"][check]
            self.assertFalse(correctness.passed)
            self.assertIn("sigma", correctness.detail)

    def test_wrong_architecture_fails_closed_before_materialization(self) -> None:
        source_module = _FakeSourceModule()

        with self.assertRaises(vc.PatchCampaignError):
            vc.run_rd04_contract_correctness(
                base_revision="a" * 40,
                hip_path=Path("H:/hip"),
                amdgpu_targets="gfx1151",
                worktree_root=Path("W:/worktrees"),
                build_root=Path("B:/build"),
                model=Path("M:/tierA-qwen4b-q6k.gguf"),
                corpus=Path("C:/wikitext2/test.txt"),
                run_dir=Path(tempfile.mkdtemp()),
                _source_module=source_module,
            )

        self.assertEqual(source_module.resolve_focals, [])

    def test_multi_architecture_build_fails_closed_before_materialization(self) -> None:
        source_module = _FakeSourceModule()

        with self.assertRaises(vc.PatchCampaignError):
            vc.run_rd04_contract_correctness(
                base_revision="a" * 40,
                hip_path=Path("H:/hip"),
                amdgpu_targets="gfx1100;gfx1201",
                worktree_root=Path("W:/worktrees"),
                build_root=Path("B:/build"),
                model=Path("M:/tierA-qwen4b-q6k.gguf"),
                corpus=Path("C:/wikitext2/test.txt"),
                run_dir=Path(tempfile.mkdtemp()),
                _source_module=source_module,
            )

        self.assertEqual(source_module.resolve_focals, [])


if __name__ == "__main__":
    unittest.main()
