"""RD13 backend_reference orchestration tests -- hardware-free."""

from __future__ import annotations

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
    build_dir, *, source_root, architecture, binary, requested_cmake_args,
    build_env, extra_binaries=(),
):
    return _FakeBuildEvidence()


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


class _FakeAttestation:
    def __init__(self, arm: str) -> None:
        self.arm = arm

    def document(self) -> dict[str, object]:
        return {"schema_version": 1, "backend": "ROCm", "arm": self.arm}


class _FakeSession:
    def __init__(self, **kwargs) -> None:
        self.binary = Path(kwargs["binary"])
        self.arm = "control" if "control" in str(self.binary) else "subject"
        self.attestation = None
        self.base_url = "http://unused.invalid"

    def __enter__(self):
        self.attestation = _FakeAttestation(self.arm)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        pass


def _row(generated_id: int, values: list[float]) -> dict[str, object]:
    return {
        "id": generated_id,
        "top_logprobs": [
            {"id": token_id, "logprob": value}
            for token_id, value in enumerate(values)
        ],
    }


def _streamer(control_rows, subject_rows):
    def stream(session, payload, timeout_s):
        if payload["stream"] is not True:
            raise AssertionError("RD13 backend_reference must use streaming completion")
        if payload["post_sampling_probs"] is not False:
            raise AssertionError("RD13 backend_reference must compare pre-sampling logprobs")
        if payload["n_probs"] != 3:
            raise AssertionError("test expected full fake vocabulary")
        return iter(control_rows if session.arm == "control" else subject_rows)

    return stream


class RunRd13BackendReferenceCheckTests(unittest.TestCase):
    def setUp(self) -> None:
        self._real_build_tree = vc.build_tree
        self._real_capture = vc.capture_completed_build_evidence
        vc.build_tree = _fake_build_tree
        vc.capture_completed_build_evidence = _fake_capture

    def tearDown(self) -> None:
        vc.build_tree = self._real_build_tree
        vc.capture_completed_build_evidence = self._real_capture

    def _run(self, *, control_rows, subject_rows):
        run_dir = Path(tempfile.mkdtemp())
        source_module = _FakeSourceModule()
        result = vc.run_rd13_backend_reference_check(
            base_revision="a" * 40,
            hip_path=Path("H:/hip"),
            amdgpu_targets="gfx1100",
            worktree_root=Path("W:/worktrees"),
            build_root=Path("B:/build"),
            model=Path("M:/tierA-qwen4b-q6k.gguf"),
            run_dir=run_dir,
            vocab_size=3,
            n_predict=2,
            tolerance=0.0005,
            _session_factory=_FakeSession,
            _stream_request=_streamer(control_rows, subject_rows),
            _source_module=source_module,
        )
        return result, run_dir, source_module

    def test_within_tolerance_passes_and_binds_full_vocab_summary(self) -> None:
        result, run_dir, source_module = self._run(
            control_rows=[
                _row(1, [-1.0, -2.0, -3.0]),
                _row(2, [-1.1, -2.1, -3.1]),
            ],
            subject_rows=[
                _row(1, [-1.0001, -2.0, -3.0]),
                _row(2, [-1.1, -2.1002, -3.1]),
            ],
        )
        correctness = result["results"]["backend_reference"]
        self.assertTrue(correctness.passed)
        self.assertEqual(result["comparison"]["logprobs_compared"], 6)
        self.assertLess(result["comparison"]["max_abs_logprob_diff"], 0.0005)
        self.assertTrue((run_dir / "artifacts" / "rd13-backend-reference.json").exists())
        self.assertEqual(
            source_module.resolve_focals,
            [None, "1206_rd13_mul_mat_add_view_fusion"],
        )

    def test_numeric_delta_over_tolerance_is_a_correctness_failure(self) -> None:
        result, _, _ = self._run(
            control_rows=[
                _row(1, [-1.0, -2.0, -3.0]),
                _row(2, [-1.0, -2.0, -3.0]),
            ],
            subject_rows=[
                _row(1, [-1.0, -2.001, -3.0]),
                _row(2, [-1.0, -2.0, -3.0]),
            ],
        )
        correctness = result["results"]["backend_reference"]
        self.assertFalse(correctness.passed)
        self.assertAlmostEqual(result["comparison"]["max_abs_logprob_diff"], 0.001)
        self.assertIn("tolerance", correctness.detail)

    def test_generated_token_divergence_is_a_correctness_failure(self) -> None:
        result, _, _ = self._run(
            control_rows=[
                _row(1, [-1.0, -2.0, -3.0]),
                _row(2, [-1.0, -2.0, -3.0]),
            ],
            subject_rows=[
                _row(2, [-1.0, -2.0, -3.0]),
                _row(2, [-1.0, -2.0, -3.0]),
            ],
        )
        correctness = result["results"]["backend_reference"]
        self.assertFalse(correctness.passed)
        self.assertEqual(result["comparison"]["decode_steps_compared"], 1)
        self.assertEqual(result["comparison"]["first_generated_token_mismatch"]["step"], 0)

    def test_incomplete_full_vocab_response_fails_closed(self) -> None:
        with self.assertRaises(vc.PatchCampaignError):
            self._run(
                control_rows=[
                    _row(1, [-1.0, -2.0]),
                    _row(2, [-1.0, -2.0, -3.0]),
                ],
                subject_rows=[
                    _row(1, [-1.0, -2.0, -3.0]),
                    _row(2, [-1.0, -2.0, -3.0]),
                ],
            )


if __name__ == "__main__":
    unittest.main()
