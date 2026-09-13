from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest


TOOLS_ROOT = Path(__file__).resolve().parents[2]
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from bigcherry.patch import validation_campaign as vc


TRACE_MARKER = "BIGCHERRY_PATCH_HIT patch=1205_rd12 path=dual_output_mmvq_fusion"

EVIDENCE_PATCHES = (
    "1222_hi67_deterministic_test_backend_ops_seed",
    "1223_hi67_machine_readable_correctness_metrics",
    "1258_rd12_paired_mul_mat_test_case",
)

SUBJECT_PATCH = "1205_rd12_paired_mmvq_dual_output"


class _EvidenceError(RuntimeError):
    pass


class _FakeBuildEvidence:
    def __init__(self, identity: str) -> None:
        self.identity = identity

    def campaign_identity(self) -> dict[str, object]:
        return {"identity": self.identity}


def _fake_build_tree(*, name, hip_path, amdgpu_targets, workdir, targets, source, extra_cmake_args):
    if targets != ["test-backend-ops"]:
        raise AssertionError(targets)
    return Path("/fake") / name / "bin"


def _fake_capture(build_dir, *, source_root, architecture, binary, requested_cmake_args, build_env):
    return _FakeBuildEvidence(f"{build_dir}:{binary}")


class _FakeSourceModule:
    REPO_ROOT = Path("/repo")

    def __init__(self) -> None:
        self.resolve_extra_patches: list[tuple[str, ...]] = []
        self.materialized: list[Path] = []

    def resolve_source_composition(self, source, *, extra_patches=(), focal=None, base_ref, base_repo):
        composition = tuple(extra_patches)
        self.resolve_extra_patches.append(composition)
        return base_ref, composition

    def materialize_composition(
        self, *, base_repo, worktree_root, resolved_revision, composition, overlay_root, requested_revision,
    ):
        path = Path(worktree_root) / "tree"
        self.materialized.append(path)
        return path

    def git_worktree_tree(self, path):
        return f"tree:{Path(path)}"


class _FakeRunner:
    def __init__(self, *, subject_marker: bool = True, control_marker: bool = False) -> None:
        self.subject_marker = subject_marker
        self.control_marker = control_marker
        self.calls: list[tuple[str, dict[str, str]]] = []

    def __call__(self, argv, **kwargs):
        env = dict(kwargs.get("env") or {})
        executable = str(argv[0])

        if "rd12-correctness-subject" in executable:
            arm = "subject"
            emit_marker = self.subject_marker
        elif "rd12-correctness-control" in executable:
            arm = "control"
            emit_marker = self.control_marker
        else:
            arm = "unknown"
            emit_marker = False

        self.calls.append((arm, env))

        return SimpleNamespace(
            returncode=0, stdout="",
            stderr=(TRACE_MARKER + "\n" if emit_marker else ""),
        )


class _FakeEvidenceModule:
    def __init__(
        self, *,
        output_mismatch: tuple[str, int] | None = None,
        reference_mismatch: tuple[str, int] | None = None,
        nels_mismatch: tuple[str, int] | None = None,
        backend_failure: tuple[str, int] | None = None,
        malformed: tuple[str, int] | None = None,
    ) -> None:
        self.output_mismatch = output_mismatch
        self.reference_mismatch = reference_mismatch
        self.nels_mismatch = nels_mismatch
        self.backend_failure = backend_failure
        self.malformed = malformed
        self.calls: list[tuple[str, str, int]] = []

    def collect_native_seed_evidence(
        self, binary, *, op_filter=None, test_file=None, moe_glu_file=None,
        target_tensor, digest_tensor, seed, runner, **kwargs,
    ):
        self.assert_probe_shape(
            op_filter=op_filter, test_file=test_file, moe_glu_file=moe_glu_file,
            target_tensor=target_tensor, digest_tensor=digest_tensor,
        )

        binary_s = str(binary)
        if "rd12-correctness-subject" in binary_s:
            arm = "subject"
        elif "rd12-correctness-control" in binary_s:
            arm = "control"
        else:
            raise AssertionError(binary_s)

        self.calls.append((arm, target_tensor, seed))

        key = (target_tensor, seed)
        if self.malformed == key:
            raise _EvidenceError("malformed correctness evidence")

        runner(
            [str(binary)], capture_output=True, text=True,
            env={"BIGCHERRY_TEST_DETERMINISTIC_SEED": str(seed)},
        )

        reference_output_digest = f"cpu-{target_tensor}-{seed}"
        native_output_digest = f"gpu-{target_tensor}-{seed}"
        output_nels = 1024

        if arm == "subject" and self.reference_mismatch == key:
            reference_output_digest += "-subject"
        if arm == "subject" and self.output_mismatch == key:
            native_output_digest += "-subject"
        if arm == "subject" and self.nels_mismatch == key:
            output_nels += 1

        nmse = 1e-6
        threshold = 1e-3
        if arm == "subject" and self.backend_failure == key:
            nmse = 1.0

        return SimpleNamespace(
            seed=seed, reference_digest=f"input-{seed}", e_n_nmse=nmse,
            max_abs_native=0.25, threshold_t=threshold,
            native_execution_status="ok",
            native_output_digest=native_output_digest,
            reference_output_digest=reference_output_digest,
            output_nels=output_nels,
        )

    @staticmethod
    def assert_probe_shape(*, op_filter, test_file, moe_glu_file, target_tensor, digest_tensor):
        if op_filter != "bigcherry_rd12=1":
            raise AssertionError(op_filter)
        if test_file is not None:
            raise AssertionError("RD12 must not use single-op --test-file")
        if moe_glu_file is not None:
            raise AssertionError(moe_glu_file)
        if target_tensor not in {"rd12_k_out", "rd12_v_out"}:
            raise AssertionError(target_tensor)
        if digest_tensor != "rd12_x":
            raise AssertionError(digest_tensor)


class RD12CorrectnessCampaignTests(unittest.TestCase):
    def setUp(self) -> None:
        self._old_build_tree = vc.build_tree
        self._old_capture = vc.capture_completed_build_evidence
        vc.build_tree = _fake_build_tree
        vc.capture_completed_build_evidence = _fake_capture

    def tearDown(self) -> None:
        vc.build_tree = self._old_build_tree
        vc.capture_completed_build_evidence = self._old_capture

    def _run(self, *, architecture: str = "gfx1100", evidence=None, runner=None):
        source = _FakeSourceModule()
        evidence = evidence or _FakeEvidenceModule()
        runner = runner or _FakeRunner()

        temp_root = Path(tempfile.mkdtemp())
        run_dir = temp_root / "run"
        run_dir.mkdir(parents=True, exist_ok=True)

        result = vc.run_rd12_correctness_check(
            base_revision="a" * 40, hip_path=Path("/opt/rocm"),
            amdgpu_targets=architecture,
            worktree_root=temp_root / "worktrees", build_root=temp_root / "build",
            build_env={"HIP_PATH": "/opt/rocm"}, run_dir=run_dir,
            _source_module=source, _evidence_module=evidence, _runner=runner,
        )
        return result, run_dir, source, evidence, runner

    def test_exact_gate_passes_on_all_contract_architectures(self) -> None:
        for architecture in ("gfx1100", "gfx1201", "gfx1030"):
            with self.subTest(architecture=architecture):
                result, run_dir, source, evidence, runner = self._run(architecture=architecture)

                self.assertTrue(result["results"]["bit_identical"].passed)
                self.assertTrue(result["results"]["backend_reference"].passed)
                self.assertTrue(result["results"]["activation"].passed)

                self.assertEqual(
                    source.resolve_extra_patches,
                    [EVIDENCE_PATCHES, (*EVIDENCE_PATCHES, SUBJECT_PATCH)],
                )
                self.assertNotIn(
                    "1236_hi105_deterministic_mul_mat_id_ids", EVIDENCE_PATCHES,
                )

                self.assertEqual(len(evidence.calls), 12)
                self.assertEqual(len(runner.calls), 12)
                self.assertEqual(len(result["rows"]), 6)

                for row in result["rows"]:
                    self.assertTrue(row["reference_equal"])
                    self.assertTrue(row["output_equal"])
                    self.assertTrue(row["nels_equal"])
                    self.assertTrue(row["bit_identical"])

                doc = json.loads(
                    (run_dir / "artifacts" / "rd12-correctness.json").read_text(encoding="utf-8")
                )
                self.assertTrue(doc["passed"])
                self.assertEqual(doc["architecture"], architecture)
                self.assertEqual(doc["evidence_patches"], list(EVIDENCE_PATCHES))
                self.assertEqual(doc["subject_patch"], SUBJECT_PATCH)
                self.assertEqual(doc["shape"]["m"], 1024)
                self.assertEqual(doc["shape"]["n"], 1)
                self.assertEqual(doc["shape"]["k"], 2560)
                self.assertEqual(len(doc["rows"]), 6)
                self.assertTrue(doc["activation"]["passed"])

    def test_subject_gpu_digest_difference_fails_bit_identical(self) -> None:
        evidence = _FakeEvidenceModule(output_mismatch=("rd12_v_out", 2))
        result, _, _, _, _ = self._run(evidence=evidence)

        self.assertFalse(result["results"]["bit_identical"].passed)
        self.assertTrue(result["results"]["backend_reference"].passed)
        self.assertTrue(result["results"]["activation"].passed)
        self.assertIn("lane='v'", result["results"]["bit_identical"].detail)
        self.assertIn("seed=2", result["results"]["bit_identical"].detail)
        self.assertIn("output_equal=False", result["results"]["bit_identical"].detail)

    def test_cpu_reference_digest_difference_fails_bit_identical(self) -> None:
        evidence = _FakeEvidenceModule(reference_mismatch=("rd12_k_out", 1))
        result, _, _, _, _ = self._run(evidence=evidence)

        self.assertFalse(result["results"]["bit_identical"].passed)
        self.assertIn("reference_equal=False", result["results"]["bit_identical"].detail)

    def test_output_element_count_difference_fails_bit_identical(self) -> None:
        evidence = _FakeEvidenceModule(nels_mismatch=("rd12_v_out", 3))
        result, _, _, _, _ = self._run(evidence=evidence)

        self.assertFalse(result["results"]["bit_identical"].passed)
        self.assertIn("nels_equal=False", result["results"]["bit_identical"].detail)

    def test_backend_reference_is_independent_of_exact_output_gate(self) -> None:
        evidence = _FakeEvidenceModule(backend_failure=("rd12_k_out", 3))
        result, _, _, _, _ = self._run(evidence=evidence)

        self.assertTrue(result["results"]["bit_identical"].passed)
        self.assertFalse(result["results"]["backend_reference"].passed)
        self.assertTrue(result["results"]["activation"].passed)
        self.assertIn("seed=3", result["results"]["backend_reference"].detail)

    def test_missing_subject_activation_marker_fails_closed(self) -> None:
        runner = _FakeRunner(subject_marker=False)
        result, _, _, _, _ = self._run(runner=runner)

        self.assertFalse(result["results"]["activation"].passed)
        self.assertFalse(result["results"]["bit_identical"].passed)
        self.assertIn(
            "focal dual-output MMVQ path was not proven active",
            result["results"]["bit_identical"].detail,
        )

    def test_control_activation_marker_fails_closed(self) -> None:
        runner = _FakeRunner(control_marker=True)
        result, _, _, _, _ = self._run(runner=runner)

        self.assertFalse(result["results"]["activation"].passed)
        self.assertFalse(result["results"]["bit_identical"].passed)

    def test_unsupported_or_multi_architecture_fails_before_resolution(self) -> None:
        for architecture in ("gfx1151", "gfx1100;gfx1201", ""):
            with self.subTest(architecture=architecture):
                source = _FakeSourceModule()

                with self.assertRaisesRegex(vc.PatchCampaignError, "exactly one contract architecture"):
                    vc.run_rd12_correctness_check(
                        base_revision="a" * 40, hip_path=Path("/opt/rocm"),
                        amdgpu_targets=architecture,
                        worktree_root=Path(tempfile.mkdtemp()) / "worktrees",
                        build_root=Path(tempfile.mkdtemp()) / "build",
                        build_env={}, run_dir=Path(tempfile.mkdtemp()),
                        _source_module=source, _evidence_module=_FakeEvidenceModule(),
                        _runner=_FakeRunner(),
                    )

                self.assertEqual(source.resolve_extra_patches, [])

    def test_malformed_backend_ops_evidence_is_hard_campaign_error(self) -> None:
        evidence = _FakeEvidenceModule(malformed=("rd12_k_out", 1))

        with self.assertRaises(_EvidenceError):
            self._run(evidence=evidence)


if __name__ == "__main__":
    unittest.main()
