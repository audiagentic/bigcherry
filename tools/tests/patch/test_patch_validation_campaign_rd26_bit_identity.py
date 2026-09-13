"""RD26 decode-vs-verify raw-logit orchestration tests -- hardware-free."""

from __future__ import annotations

import json
import subprocess
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
    if targets != ["llama-results"]:
        raise AssertionError("RD26 bit-identity producer must build llama-results")
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
        self.resolve_extra_patches: list[tuple[str, ...]] = []

    def resolve_source_composition(self, source, *, extra_patches=(), focal=None, base_ref, base_repo):
        if focal is not None:
            raise AssertionError("RD26 bit identity must use explicit whole-composition arms")
        patches = tuple(extra_patches)
        self.resolve_extra_patches.append(patches)
        return base_ref, patches

    def materialize_composition(
        self, *, base_repo, worktree_root, resolved_revision, composition, overlay_root, requested_revision,
    ):
        return Path(worktree_root) / "tree"

    def git_worktree_tree(self, source):
        return f"tree:{source}"


class _FakeRunner:
    def __init__(
        self, *,
        control_diverges: bool = True,
        subject_diverges: bool = False,
        nondeterministic: tuple[str, str] | None = None,
        fail: tuple[str, str] | None = None,
        omit_output: tuple[str, str] | None = None,
    ) -> None:
        self.control_diverges = control_diverges
        self.subject_diverges = subject_diverges
        self.nondeterministic = nondeterministic
        self.fail = fail
        self.omit_output = omit_output
        self.counts: dict[tuple[str, str], int] = {}
        self.calls: list[dict[str, object]] = []

    def __call__(self, argv, **kwargs):
        args = [str(value) for value in argv]
        binary = args[0]
        arm = "subject" if "subject" in binary else "control"
        ubatch = int(args[args.index("--ubatch-size") + 1])
        mode = "decode" if ubatch == 1 else "verify"
        key = (arm, mode)
        replicate = self.counts.get(key, 0)
        self.counts[key] = replicate + 1

        self.calls.append({
            "arm": arm, "mode": mode, "ubatch": ubatch,
            "batch": int(args[args.index("--batch-size") + 1]),
            "ctx": int(args[args.index("--ctx-size") + 1]),
        })

        if self.fail == key:
            return subprocess.CompletedProcess(args, 7, "", "synthetic failure")

        output = Path(args[args.index("--output") + 1])

        if self.omit_output == key:
            return subprocess.CompletedProcess(args, 0, "", "")

        if arm == "control":
            marker = b"B" if (self.control_diverges and mode == "verify") else b"A"
        else:
            marker = b"D" if (self.subject_diverges and mode == "verify") else b"C"

        data = b"GGUF" + marker * 64

        if self.nondeterministic == key and replicate > 0:
            data = data[:-1] + b"Z"

        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(data)

        return subprocess.CompletedProcess(args, 0, "ok", "")


class RunRd26DecodeVerifyBitIdentityCheckTests(unittest.TestCase):
    SUBJECT_PATCH = "1210_rd26_bitidentical_decode_verify_standalone"

    def setUp(self) -> None:
        self._real_build_tree = vc.build_tree
        self._real_capture = vc.capture_completed_build_evidence
        self._real_parity = vc.assert_validation_subject_parity
        vc.build_tree = _fake_build_tree
        vc.capture_completed_build_evidence = _fake_capture
        vc.assert_validation_subject_parity = _fake_parity

    def tearDown(self) -> None:
        vc.build_tree = self._real_build_tree
        vc.capture_completed_build_evidence = self._real_capture
        vc.assert_validation_subject_parity = self._real_parity

    def _run(self, *, runner: _FakeRunner | None = None, spec_draft_n_max: int = 4):
        run_dir = Path(tempfile.mkdtemp())
        source = _FakeSourceModule()
        runner = runner or _FakeRunner()

        result = vc.run_rd26_decode_verify_bit_identity_check(
            base_revision="a" * 40, hip_path=Path("H:/hip"), amdgpu_targets="gfx1100",
            worktree_root=Path("W:/worktrees"), build_root=Path("B:/build"),
            model=Path("M:/tierA-qwen4b-q6k.gguf"), run_dir=run_dir, build_env={},
            spec_draft_n_max=spec_draft_n_max, ctx_size=64,
            prompt="one two three four five six seven eight nine ten eleven twelve",
            replicates=2, _source_module=source, _runner=runner,
        )
        return result, run_dir, source, runner

    def test_subject_identity_and_control_divergence_pass_nonvacuously(self) -> None:
        result, run_dir, source, runner = self._run()

        correctness = result["results"]["bit_identical"]
        self.assertTrue(correctness.passed)
        self.assertTrue(result["comparison"]["subject_bit_identical"])
        self.assertTrue(result["comparison"]["control_diverged"])
        self.assertEqual(result["comparison"]["decode_ubatch"], 1)
        self.assertEqual(result["comparison"]["verify_ubatch"], 5)

        self.assertEqual(
            source.resolve_extra_patches, [(), (self.SUBJECT_PATCH,)],
        )

        self.assertEqual(len(runner.calls), 8)
        self.assertEqual({call["ubatch"] for call in runner.calls}, {1, 5})
        self.assertEqual({call["batch"] for call in runner.calls}, {64})
        self.assertEqual({call["ctx"] for call in runner.calls}, {64})

        artifact = run_dir / "artifacts" / "rd26-decode-verify-bit-identity.json"
        self.assertTrue(artifact.exists())

        doc = json.loads(artifact.read_text(encoding="utf-8"))
        self.assertTrue(doc["passed"])
        self.assertTrue(doc["comparison"]["subject_bit_identical"])
        self.assertTrue(doc["comparison"]["control_diverged"])

        self.assertFalse((run_dir / "scratch" / "rd26-bit-identity").exists())

    def test_subject_divergence_fails_even_when_control_diverges(self) -> None:
        result, _, _, _ = self._run(runner=_FakeRunner(subject_diverges=True))

        correctness = result["results"]["bit_identical"]
        self.assertFalse(correctness.passed)
        self.assertFalse(result["comparison"]["subject_bit_identical"])
        self.assertTrue(result["comparison"]["control_diverged"])
        self.assertIn("subject decode/verify", correctness.detail)

    def test_control_identity_fails_nonvacuous_effect_gate(self) -> None:
        result, _, _, _ = self._run(runner=_FakeRunner(control_diverges=False))

        correctness = result["results"]["bit_identical"]
        self.assertFalse(correctness.passed)
        self.assertTrue(result["comparison"]["subject_bit_identical"])
        self.assertFalse(result["comparison"]["control_diverged"])
        self.assertIn("control is also bit-identical", correctness.detail)

    def test_same_configuration_nondeterminism_is_hard_error(self) -> None:
        with self.assertRaisesRegex(vc.PatchCampaignError, "subject/verify is not repeatable"):
            self._run(runner=_FakeRunner(nondeterministic=("subject", "verify")))

    def test_llama_results_process_failure_is_hard_error(self) -> None:
        with self.assertRaisesRegex(vc.PatchCampaignError, "control/decode/rep0.*exit 7"):
            self._run(runner=_FakeRunner(fail=("control", "decode")))

    def test_missing_output_is_hard_error(self) -> None:
        with self.assertRaisesRegex(vc.PatchCampaignError, "did not create its GGUF output"):
            self._run(runner=_FakeRunner(omit_output=("control", "decode")))

    def test_verify_width_over_rd26_scope_fails_before_source_resolution(self) -> None:
        run_dir = Path(tempfile.mkdtemp())
        source = _FakeSourceModule()

        with self.assertRaisesRegex(vc.PatchCampaignError, r"spec_draft_n_max must be in \[1, 7\]"):
            vc.run_rd26_decode_verify_bit_identity_check(
                base_revision="a" * 40, hip_path=Path("H:/hip"), amdgpu_targets="gfx1100",
                worktree_root=Path("W:/worktrees"), build_root=Path("B:/build"),
                model=Path("M:/tierA-qwen4b-q6k.gguf"), run_dir=run_dir, build_env={},
                spec_draft_n_max=8, _source_module=source, _runner=_FakeRunner(),
            )

        self.assertEqual(source.resolve_extra_patches, [])


if __name__ == "__main__":
    unittest.main()
