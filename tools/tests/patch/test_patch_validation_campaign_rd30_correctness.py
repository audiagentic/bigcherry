"""RD30 correctness orchestration tests -- hardware-free."""

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
    def __init__(self, tag: str) -> None:
        self.effective_build_id = f"build-{tag}"
        self.effective_configure = {
            "CMAKE_BUILD_TYPE": "Release",
            "GGML_HIP": "ON",
        }
        self.verification = SimpleNamespace(to_dict=lambda: {})
        self.runtime_artifacts = {}

    def campaign_identity(self) -> dict[str, object]:
        return {"effective_build_id": self.effective_build_id}


def _fake_build_tree(
    *,
    name,
    hip_path,
    amdgpu_targets,
    workdir,
    targets,
    source,
    extra_cmake_args,
):
    return Path(f"/fake/{name}/bin")


def _fake_capture(
    build_dir,
    *,
    source_root,
    architecture,
    binary,
    requested_cmake_args,
    build_env,
    extra_binaries=(),
):
    return _FakeBuildEvidence(str(binary).replace("/", "_"))


class _FakeSourceModule:
    REPO_ROOT = Path("R:/repo")

    def __init__(self) -> None:
        self.resolve_extra_patches: list[tuple[str, ...]] = []

    def resolve_source_composition(
        self,
        source,
        *,
        extra_patches=(),
        focal=None,
        base_ref,
        base_repo,
    ):
        if focal is not None:
            raise AssertionError(
                "RD30 must use explicit whole-composition arms"
            )
        patches = tuple(extra_patches)
        self.resolve_extra_patches.append(patches)
        return base_ref, patches

    def materialize_composition(
        self,
        *,
        base_repo,
        worktree_root,
        resolved_revision,
        composition,
        overlay_root,
        requested_revision,
    ):
        return Path(worktree_root) / "tree"

    def git_worktree_tree(self, source):
        return f"tree:{source}"


class _FakeSignatureModule:
    def __init__(self) -> None:
        self.signatures: list[dict[str, object]] = []

    @staticmethod
    def load_ggml_op_names(vendor_root):
        return {17: "MUL_MAT_ID"}

    @staticmethod
    def load_ggml_type_names(vendor_root):
        return {
            0: "F32",
            12: "Q4_K",
            13: "Q8_0",
        }

    def signature_to_mul_mat_id_test_file_line(
        self,
        signature,
        *,
        vendor_root,
    ):
        self.signatures.append(dict(signature))
        return (
            f"fake-test-line src0_type={signature['src0_type']}",
            "out",
            "leaf_2",
        )


class _EvidenceError(RuntimeError):
    pass


class _FakeEvidenceModule:
    EvidenceError = _EvidenceError

    def __init__(
        self,
        *,
        output_mismatch: tuple[str, int] | None = None,
        routing_mismatch: tuple[str, int] | None = None,
        backend_failure: tuple[str, int] | None = None,
        malformed: tuple[str, int] | None = None,
    ) -> None:
        self.output_mismatch = output_mismatch
        self.routing_mismatch = routing_mismatch
        self.backend_failure = backend_failure
        self.malformed = malformed
        self.calls: list[tuple[str, str, int]] = []

    def collect_native_seed_evidence(
        self,
        binary,
        *,
        op_filter=None,
        test_file=None,
        moe_glu_file=None,
        target_tensor,
        digest_tensor=None,
        seed,
        env=None,
        runner=None,
    ):
        if (
            op_filter is not None
            or moe_glu_file is not None
            or test_file is None
        ):
            raise AssertionError(
                "RD30 must use exact --test-file MUL_MAT_ID cases"
            )
        if target_tensor != "out" or digest_tensor != "leaf_2":
            raise AssertionError(
                "unexpected RD30 target/digest tensor"
            )

        arm = "subject" if "subject" in str(binary) else "control"
        shape = Path(test_file).stem
        self.calls.append((arm, shape, seed))

        if self.malformed == (shape, seed) and arm == "subject":
            raise self.EvidenceError(
                "missing BIGCHERRY_CORRECTNESS_METRIC"
            )

        ids_digest = f"ids-{shape}-{seed}"
        if (
            self.routing_mismatch == (shape, seed)
            and arm == "subject"
        ):
            ids_digest += "-different"

        output_digest = f"gpu-{shape}-{seed}"
        if (
            self.output_mismatch == (shape, seed)
            and arm == "subject"
        ):
            output_digest += "-different"

        nmse = 1e-6
        threshold = 5e-4
        if (
            self.backend_failure == (shape, seed)
            and arm == "subject"
        ):
            nmse = 1e-2

        return SimpleNamespace(
            seed=seed,
            reference_digest=ids_digest,
            e_n_nmse=nmse,
            max_abs_native=1e-5,
            threshold_t=threshold,
            native_execution_status="ok",
            native_output_digest=output_digest,
            reference_output_digest=f"cpu-{shape}-{seed}",
            output_nels=256 * 8 * 32,
        )


def _unused_runner(*args, **kwargs):
    raise AssertionError(
        "fake evidence module must not execute hardware"
    )


class RunRd30CorrectnessCheckTests(unittest.TestCase):
    SUPPORT = (
        "1222_hi67_deterministic_test_backend_ops_seed",
        "1223_hi67_machine_readable_correctness_metrics",
        "1236_hi105_deterministic_mul_mat_id_ids",
    )

    def setUp(self) -> None:
        self._real_build_tree = vc.build_tree
        self._real_capture = vc.capture_completed_build_evidence
        vc.build_tree = _fake_build_tree
        vc.capture_completed_build_evidence = _fake_capture

    def tearDown(self) -> None:
        vc.build_tree = self._real_build_tree
        vc.capture_completed_build_evidence = self._real_capture

    def _run(
        self,
        *,
        evidence: _FakeEvidenceModule | None = None,
        amdgpu_targets: str = "gfx1100",
    ):
        run_dir = Path(tempfile.mkdtemp())
        source = _FakeSourceModule()
        signatures = _FakeSignatureModule()
        evidence = evidence or _FakeEvidenceModule()

        result = vc.run_rd30_correctness_check(
            base_revision="a" * 40,
            hip_path=Path("H:/hip"),
            amdgpu_targets=amdgpu_targets,
            worktree_root=Path("W:/worktrees"),
            build_root=Path("B:/build"),
            build_env={},
            run_dir=run_dir,
            _source_module=source,
            _evidence_module=evidence,
            _signature_module=signatures,
            _runner=_unused_runner,
        )
        return result, run_dir, source, signatures, evidence

    def test_passes_exact_digest_gate_for_two_256_expert_shapes_and_three_seeds(
        self,
    ) -> None:
        result, run_dir, source, signatures, evidence = self._run()

        self.assertTrue(
            result["results"]["bit_identical"].passed
        )
        self.assertTrue(
            result["results"]["backend_reference"].passed
        )
        self.assertEqual(len(result["rows"]), 6)
        self.assertEqual(
            len(evidence.calls),
            12,
        )  # control + subject per row
        self.assertTrue(
            (
                run_dir
                / "artifacts"
                / "rd30-correctness.json"
            ).exists()
        )

        self.assertEqual(
            source.resolve_extra_patches,
            [
                self.SUPPORT,
                (
                    *self.SUPPORT,
                    "1237_rd30_moe_mmq_compact_grid",
                ),
            ],
        )

        self.assertEqual(len(signatures.signatures), 2)
        for signature in signatures.signatures:
            self.assertEqual(signature["flags"], 0x0F)
            self.assertEqual(
                signature["ne0"],
                [2048, 256, 256, 1],
            )
            self.assertEqual(
                signature["ne1"],
                [2048, 1, 32, 1],
            )
            self.assertEqual(
                signature["ned"],
                [256, 8, 32, 1],
            )
            self.assertEqual(
                signature["n_expert"],
                256,
            )
            self.assertEqual(
                signature["n_expert_used"],
                8,
            )

        doc = json.loads(
            (
                run_dir
                / "artifacts"
                / "rd30-correctness.json"
            ).read_text(encoding="utf-8")
        )
        self.assertTrue(doc["passed"])
        self.assertEqual(
            doc["architecture"],
            "gfx1100",
        )
        self.assertEqual(len(doc["rows"]), 6)

    def test_single_subject_output_digest_difference_fails_bit_identical(
        self,
    ) -> None:
        evidence = _FakeEvidenceModule(
            output_mismatch=(
                "q8_0-moe-prefill32",
                2,
            )
        )
        result, run_dir, _, _, _ = self._run(
            evidence=evidence
        )

        self.assertFalse(
            result["results"]["bit_identical"].passed
        )
        self.assertTrue(
            result["results"]["backend_reference"].passed
        )
        self.assertIn(
            "q8_0-moe-prefill32",
            result["results"]["bit_identical"].detail,
        )
        self.assertIn(
            "seed=2",
            result["results"]["bit_identical"].detail,
        )

        doc = json.loads(
            (
                run_dir
                / "artifacts"
                / "rd30-correctness.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(len(doc["rows"]), 6)
        self.assertEqual(
            sum(
                not row["bit_identical"]
                for row in doc["rows"]
            ),
            1,
        )

    def test_routing_digest_difference_fails_even_when_outputs_match(
        self,
    ) -> None:
        evidence = _FakeEvidenceModule(
            routing_mismatch=(
                "q4_k-moe-prefill32",
                1,
            )
        )
        result, _, _, _, _ = self._run(
            evidence=evidence
        )

        self.assertFalse(
            result["results"]["bit_identical"].passed
        )
        self.assertIn(
            "ids_equal=False",
            result["results"]["bit_identical"].detail,
        )

    def test_backend_reference_is_reported_independently_from_exact_output_gate(
        self,
    ) -> None:
        evidence = _FakeEvidenceModule(
            backend_failure=(
                "q4_k-moe-prefill32",
                3,
            )
        )
        result, _, _, _, _ = self._run(
            evidence=evidence
        )

        self.assertTrue(
            result["results"]["bit_identical"].passed
        )
        self.assertFalse(
            result["results"]["backend_reference"].passed
        )
        self.assertIn(
            "seed=3",
            result["results"]["backend_reference"].detail,
        )

    def test_non_gfx1100_fails_before_source_materialization(
        self,
    ) -> None:
        source = _FakeSourceModule()

        with self.assertRaisesRegex(
            vc.PatchCampaignError,
            "gfx1100 exactly",
        ):
            vc.run_rd30_correctness_check(
                base_revision="a" * 40,
                hip_path=Path("H:/hip"),
                amdgpu_targets="gfx1201",
                worktree_root=Path("W:/worktrees"),
                build_root=Path("B:/build"),
                build_env={},
                run_dir=Path(tempfile.mkdtemp()),
                _source_module=source,
                _evidence_module=_FakeEvidenceModule(),
                _signature_module=_FakeSignatureModule(),
                _runner=_unused_runner,
            )

        self.assertEqual(
            source.resolve_extra_patches,
            [],
        )

    def test_malformed_backend_ops_evidence_is_a_hard_campaign_error(
        self,
    ) -> None:
        evidence = _FakeEvidenceModule(
            malformed=(
                "q4_k-moe-prefill32",
                1,
            )
        )

        with self.assertRaises(_EvidenceError):
            self._run(evidence=evidence)


if __name__ == "__main__":
    unittest.main()
