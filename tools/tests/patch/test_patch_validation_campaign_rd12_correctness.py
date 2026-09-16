"""PA36 sub-slice 2 (T10, dev-gpt-agent req_2ecda033763949a9): the RD12
bit-identical measurement moved from validation_campaign.py's deleted
run_rd12_correctness_check() into the patch-local producer at
patches/1205_rd12_paired_mmvq_dual_output/validation/producer.py.

These tests drive the REAL producer module (loaded through the real
resolve_producer() loader, so producer.toml policy is exercised too)
against a fake ProducerRuntime and a faked
collect_native_seed_evidence -- the same fake evidence/runner machinery
the deleted CLI-path tests used, adapted to the producer protocol. The
dispatcher/binder/exit-semantics side of the migration is covered by
test_patch_validation_campaign_rd12_contract_cli.py.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping, NoReturn, cast
from unittest import mock

TOOLS_ROOT = Path(__file__).resolve().parents[2]
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from bigcherry.experiment.attestation import ExecutionIdentity  # noqa: E402
from bigcherry.patch import activation as patch_activation  # noqa: E402
from bigcherry.patch import source as psi  # noqa: E402
from bigcherry.patch import validation_producer as vp  # noqa: E402
from bigcherry.patch.validation import ArtifactRef  # noqa: E402
from bigcherry.tuning import correctness_evidence  # noqa: E402

SUBJECT_PATCH = "1205_rd12_paired_mmvq_dual_output"
PATCH_DIR = TOOLS_ROOT.parent / "patches" / SUBJECT_PATCH
ARCHITECTURES = ("gfx1100", "gfx1201", "gfx1030")
FAT_TARGETS = "gfx1100;gfx1201;gfx1030"
EVIDENCE_PATCHES = (
    "1222_hi67_deterministic_test_backend_ops_seed",
    "1223_hi67_machine_readable_correctness_metrics",
    "1258_rd12_paired_mul_mat_test_case",
)
TRACE_MARKER = "BIGCHERRY_PATCH_HIT patch=1205_rd12 path=dual_output_mmvq_fusion"


class _EvidenceError(RuntimeError):
    pass


def _correctness_doc(result: vp.ProducerResult) -> dict[str, object]:
    value = result.correctness
    if not isinstance(value, Mapping):
        raise AssertionError(f"producer returned no correctness dict: {value!r}")
    return dict(value)


def _trace_map(result: vp.ProducerResult) -> Mapping[str, object]:
    trace = result.trace_evidence
    if not isinstance(trace, Mapping):
        raise AssertionError(f"producer returned no trace evidence: {trace!r}")
    return trace


def _trace_observation(result: vp.ProducerResult, role: str) -> Mapping[str, object]:
    observation = _trace_map(result)[role]
    if not isinstance(observation, Mapping):
        raise AssertionError(
            f"trace evidence {role!r} is not an object: {observation!r}"
        )
    return observation


def _artifact_ref(result: vp.ProducerResult, role: str) -> Mapping[str, object]:
    artifact = _trace_observation(result, role)["artifact"]
    if not isinstance(artifact, Mapping):
        raise AssertionError(
            f"trace evidence {role!r} artifact is not an object: {artifact!r}"
        )
    return artifact


def _activation_evidence(
    result: vp.ProducerResult,
) -> patch_activation.ActivationEvidence:
    value = result.activation_evidence
    if not isinstance(value, patch_activation.ActivationEvidence):
        raise AssertionError(f"bad activation evidence: {value!r}")
    return value


def _fake_collect(
    *,
    output_mismatch: tuple[str, int] | None = None,
    reference_mismatch: tuple[str, int] | None = None,
    nels_mismatch: tuple[str, int] | None = None,
    backend_failure: tuple[str, int] | None = None,
    malformed: tuple[str, int] | None = None,
):
    calls: list[tuple[str, str, int]] = []

    def collect_native_seed_evidence(
        binary,
        *,
        op_filter=None,
        target_tensor=None,
        digest_tensor=None,
        seed=None,
        runner=None,
        **kwargs,
    ):
        if op_filter != "bigcherry_rd12=1":
            raise AssertionError(op_filter)
        if target_tensor not in {"rd12_k_out", "rd12_v_out"}:
            raise AssertionError(target_tensor)
        if digest_tensor != "rd12_x":
            raise AssertionError(digest_tensor)
        if kwargs:
            raise AssertionError(f"unexpected probe kwargs: {kwargs!r}")
        assert seed is not None
        assert runner is not None

        binary_s = str(binary)
        if "SUBJECT-BIN" in binary_s:
            arm = "subject"
        elif "CONTROL-BIN" in binary_s:
            arm = "control"
        else:
            arm = "unknown"
        calls.append((arm, target_tensor, seed))
        key = (target_tensor, seed)
        if malformed == key:
            raise _EvidenceError("malformed correctness evidence")

        runner(
            [binary_s],
            capture_output=True,
            text=True,
            env={"BIGCHERRY_TEST_DETERMINISTIC_SEED": str(seed)},
        )

        reference_output_digest = f"cpu-{target_tensor}-{seed}"
        native_output_digest = f"gpu-{target_tensor}-{seed}"
        output_nels = 1024
        if arm == "subject" and reference_mismatch == key:
            reference_output_digest += "-subject"
        if arm == "subject" and output_mismatch == key:
            native_output_digest += "-subject"
        if arm == "subject" and nels_mismatch == key:
            output_nels += 1
        nmse = 1e-6
        if arm == "subject" and backend_failure == key:
            nmse = 1.0

        return SimpleNamespace(
            seed=seed,
            reference_digest=f"input-{seed}",
            e_n_nmse=nmse,
            max_abs_native=0.25,
            threshold_t=1e-3,
            native_execution_status="ok",
            native_output_digest=native_output_digest,
            reference_output_digest=reference_output_digest,
            output_nels=output_nels,
        )

    return collect_native_seed_evidence, calls


def _fake_subprocess_run(*, subject_marker: bool = True, control_marker: bool = False):
    """Stands in for subprocess.run inside the producer's internal
    _correctness_runner: the producer has no runner injection seam (unlike
    the deleted CLI path's _runner= point), so the seam is the stdlib call
    itself. Returns a completed-run whose stderr carries the focal marker
    per arm/flag -- the same contract the real binary honors."""
    calls: list[tuple[str, dict[str, str]]] = []

    def run(argv, **kwargs):
        env = dict(kwargs.get("env") or {})
        executable = str(argv[0])
        if "SUBJECT-BIN" in executable:
            arm, emit = "subject", subject_marker
        elif "CONTROL-BIN" in executable:
            arm, emit = "control", control_marker
        else:
            arm, emit = "unknown", False
        calls.append((arm, env))
        return SimpleNamespace(
            returncode=0,
            stdout="",
            stderr=(TRACE_MARKER + "\n" if emit else ""),
        )

    return run, calls


class _FakeRuntime:
    """ProducerRuntime seam fake: one prebuilt pair, one real
    ProducerDeviceContext, real artifact files under run_dir (the same
    ``artifacts/<name>`` bound-artifact convention as
    CampaignProducerRuntime)."""

    def __init__(
        self,
        *,
        run_dir: Path,
        pair: vp.ProducerBuildPair,
        device: vp.ProducerDeviceContext | None,
    ) -> None:
        self.run_dir = run_dir
        self.pair = pair
        self.device = device
        self.build_pair_calls: list[dict[str, object]] = []
        self.device_contexts_calls: list[Mapping[str, tuple[int, ...]]] = []

    def build_pair(
        self,
        *,
        targets: tuple[str, ...],
        primary_target: str,
        common_extra_patches: tuple[str, ...] = (),
        baseline_source: str = "bigcherry",
        control_extra_cmake_args: tuple[str, ...] = (),
        subject_extra_cmake_args: tuple[str, ...] = (),
    ) -> vp.ProducerBuildPair:
        self.build_pair_calls.append(
            {
                "targets": tuple(targets),
                "primary_target": primary_target,
                "common_extra_patches": tuple(common_extra_patches),
                "baseline_source": baseline_source,
            }
        )
        return self.pair

    def device_contexts(
        self,
        *,
        device_map: Mapping[str, tuple[int, ...]],
    ) -> tuple[vp.ProducerDeviceContext, ...]:
        self.device_contexts_calls.append(device_map)
        return (self.device,) if self.device is not None else ()

    def write_artifact(
        self,
        *,
        name: str,
        payload: Mapping[str, object],
    ) -> ArtifactRef:
        return self._write(name, json.dumps(payload, indent=2))

    def write_text_artifact(self, *, name: str, text: str) -> ArtifactRef:
        return self._write(name, text)

    def run_paired_llama_benchmark(
        self,
        *,
        control_binary: Path,
        subject_binary: Path,
        model: Path,
        workloads: tuple[str, ...] = ("decode", "prefill"),
        patch_args: tuple[str, ...] = (),
        runtime_args: tuple[str, ...] = (),
        pairs: int = 3,
        log_context: str,
        device: vp.ProducerDeviceContext | None = None,
    ) -> NoReturn:
        raise NotImplementedError("the RD12 producer never benchmarks")

    def _write(self, name: str, text: str) -> ArtifactRef:
        path = self.run_dir / "artifacts" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("utf-8"))
        return ArtifactRef(
            name=name,
            path=path.relative_to(self.run_dir).as_posix(),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )


def _make_pair(temp: Path) -> vp.ProducerBuildPair:
    return vp.ProducerBuildPair(
        base_revision="a" * 40,
        control_source=temp / "trees" / "control",
        subject_source=temp / "trees" / "subject",
        control_composition=(("1222_hi67_deterministic_test_backend_ops_seed", "c1"),),
        subject_composition=(
            ("1222_hi67_deterministic_test_backend_ops_seed", "c1"),
            (SUBJECT_PATCH, "c2"),
        ),
        control_bin=temp / "pair" / "CONTROL-BIN" / "test-backend-ops",
        subject_bin=temp / "pair" / "SUBJECT-BIN" / "test-backend-ops",
        validation_build_identities={
            "control": {"build_id": "fake-control-build"},
            "subject": {"build_id": "fake-subject-build"},
        },
    )


def _make_device(architecture: str) -> vp.ProducerDeviceContext:
    return vp.ProducerDeviceContext(
        architecture=architecture,
        device_index=0,
        execution_identity=ExecutionIdentity(
            backend="ROCm",
            architectures=(architecture,),
        ),
        env_overrides={"HIP_VISIBLE_DEVICES": "0"},
        env_unset=("ROCR_VISIBLE_DEVICES",),
    )


def _run_producer(
    *,
    architecture: str = "gfx1100",
    subject_marker: bool = True,
    control_marker: bool = False,
    with_device: bool = True,
    collect_kwargs: dict[str, tuple[str, int] | None] | None = None,
):
    temp = Path(tempfile.mkdtemp())
    run_dir = temp / "run"
    run_dir.mkdir(parents=True)
    pair = _make_pair(temp)
    device = _make_device(architecture) if with_device else None
    runtime = _FakeRuntime(run_dir=run_dir, pair=pair, device=device)
    collect, evidence_calls = _fake_collect(**(collect_kwargs or {}))
    fake_run, runner_calls = _fake_subprocess_run(
        subject_marker=subject_marker,
        control_marker=control_marker,
    )
    ctx = vp.ProducerContext(
        repo_root=TOOLS_ROOT.parent,
        patch_dir=PATCH_DIR,
        workdir=run_dir,
        campaign_id=f"{SUBJECT_PATCH}/rd12",
        base_revision="a" * 40,
        hip_path=Path("/opt/rocm"),
        fat_targets=vp.FatTargetPlan(targets=(architecture,)),
        model=None,
        corpus=None,
        build_env={"HIP_PATH": "/opt/rocm"},
        inputs={},
        validation_build_identities={},
        patch_id=SUBJECT_PATCH,
        device_map={architecture: (0,)},
        runtime=runtime,
    )
    with (
        mock.patch.object(
            correctness_evidence, "collect_native_seed_evidence", collect
        ),
        mock.patch.object(psi, "git_worktree_tree", lambda p: f"tree:{p}"),
        mock.patch("subprocess.run", fake_run),
    ):
        selection = vp.resolve_producer(patch_dir=PATCH_DIR, producer_id="rd12")
        result = selection.producer(ctx)
    return result, run_dir, runtime, pair, evidence_calls, runner_calls, selection


class RD12ProducerMeasurementTests(unittest.TestCase):
    def test_exact_gate_passes_on_all_contract_architectures(self) -> None:
        for architecture in ARCHITECTURES:
            with self.subTest(architecture=architecture):
                (
                    result,
                    run_dir,
                    runtime,
                    pair,
                    evidence_calls,
                    runner_calls,
                    selection,
                ) = _run_producer(architecture=architecture)

                # producer.toml policy is the real one, loaded for real.
                self.assertEqual(selection.spec.standard_campaign, "run")
                self.assertEqual(selection.spec.trace_probe, "skip")
                self.assertEqual(selection.spec.correctness_evidence_cli, "forbid")
                self.assertEqual(selection.spec.performance_benchmark_cli, "forbid")

                # One sanctioned pair build: fat multi-arch, evidence
                # patches shared, bigcherry baseline.
                self.assertEqual(len(runtime.build_pair_calls), 1)
                call = runtime.build_pair_calls[0]
                self.assertEqual(call["targets"], ARCHITECTURES)
                self.assertEqual(call["primary_target"], "test-backend-ops")
                self.assertEqual(call["common_extra_patches"], EVIDENCE_PATCHES)
                self.assertEqual(call["baseline_source"], "bigcherry")

                # 2 lanes x 3 seeds x 2 arms.
                self.assertEqual(len(evidence_calls), 12)
                self.assertEqual(len(runner_calls), 12)

                # Semantic correctness only -- no identity fields.
                correctness = _correctness_doc(result)
                self.assertEqual(
                    set(correctness), {"disposition", "mechanism", "detail"}
                )
                self.assertEqual(correctness["disposition"], "passed")
                self.assertEqual(
                    correctness["mechanism"],
                    "rd12-paired-mmvq-bit-identical",
                )
                self.assertIn("byte-identical", str(correctness["detail"]))

                # Contract-internal results are NOT validation-plan checks.
                self.assertEqual(result.check_results, ())
                self.assertEqual(result.lane_effects, ())
                self.assertEqual(
                    result.validation_build_identities, pair.validation_build_identities
                )

                # Trace evidence: artifact refs only, NO producer-owned
                # marker_regex (the plan owns it; the binder injects it).
                trace = _trace_map(result)
                self.assertEqual(set(trace), {"positive", "negative"})
                for role in ("positive", "negative"):
                    observation = _trace_observation(result, role)
                    self.assertEqual(set(observation), {"artifact"})
                    self.assertNotIn("marker_regex", observation)

                self.assertEqual(
                    _activation_evidence(result).status,
                    "executed",
                )
                self.assertEqual(
                    _activation_evidence(result).mechanism,
                    "rd12-trigger-marker",
                )

                # Exactly the three artifacts this architecture emits;
                # all are in the producer.toml static allowlist, and the
                # bound refs hash the real files on disk.
                expected_names = {
                    f"rd12-correctness-{architecture}.json",
                    f"activation-rd12-{architecture}-subject.log",
                    f"activation-rd12-{architecture}-control.log",
                }
                self.assertEqual(result.emitted_artifacts, frozenset(expected_names))
                for name in sorted(expected_names):
                    target = run_dir / "artifacts" / name
                    self.assertTrue(target.is_file(), name)
                    self.assertIn(name, selection.spec.artifact_names)
                for role, name in (
                    ("positive", f"activation-rd12-{architecture}-subject.log"),
                    ("negative", f"activation-rd12-{architecture}-control.log"),
                ):
                    artifact = _artifact_ref(result, role)
                    self.assertEqual(artifact["path"], f"artifacts/{name}")
                    self.assertEqual(
                        artifact["sha256"],
                        hashlib.sha256(
                            (run_dir / "artifacts" / name).read_bytes(),
                        ).hexdigest(),
                    )

                # Raw artifact document: audit-persistent contract
                # results, per-row digests, bound logs, real identities.
                doc = json.loads(
                    (
                        run_dir / "artifacts" / f"rd12-correctness-{architecture}.json"
                    ).read_text(encoding="utf-8"),
                )
                self.assertTrue(doc["passed"])
                self.assertEqual(doc["contract_id"], "RD12-PAIRED-MMVQ-DUAL")
                self.assertEqual(doc["architecture"], architecture)
                self.assertEqual(doc["compiled_targets"], FAT_TARGETS)
                self.assertEqual(doc["evidence_patches"], list(EVIDENCE_PATCHES))
                self.assertEqual(doc["subject_patch"], SUBJECT_PATCH)
                self.assertEqual(
                    (doc["shape"]["m"], doc["shape"]["n"], doc["shape"]["k"]),
                    (1024, 1, 2560),
                )
                self.assertEqual(len(doc["rows"]), 6)
                for row in doc["rows"]:
                    self.assertTrue(row["bit_identical"])
                    self.assertTrue(row["reference_equal"])
                    self.assertTrue(row["output_equal"])
                    self.assertTrue(row["nels_equal"])
                    self.assertTrue(row["backend_reference_ok"])
                self.assertEqual(
                    set(doc["contract_results"]),
                    {"bit_identical", "backend_reference", "activation"},
                )
                self.assertTrue(doc["contract_results"]["bit_identical"]["passed"])
                self.assertTrue(doc["activation"]["passed"])
                self.assertEqual(
                    doc["activation"]["subject_log"],
                    f"artifacts/activation-rd12-{architecture}-subject.log",
                )
                self.assertEqual(
                    doc["activation"]["control_log"],
                    f"artifacts/activation-rd12-{architecture}-control.log",
                )
                self.assertEqual(len(doc["activation"]["observations"]), 12)
                self.assertEqual(
                    doc["control_source_tree"],
                    f"tree:{pair.control_source}",
                )
                self.assertEqual(
                    doc["subject_source_tree"],
                    f"tree:{pair.subject_source}",
                )
                self.assertEqual(
                    doc["control_build_identity"],
                    {"build_id": "fake-control-build"},
                )
                self.assertEqual(
                    doc["subject_build_identity"],
                    {"build_id": "fake-subject-build"},
                )

                # The sanctioned HIP-only selector env, never ROCR
                # double-filtering, and the focal trace flag always on.
                for arm, env in runner_calls:
                    self.assertEqual(env["BIGCHERRY_PATCH_TRACE"], "1")
                    self.assertEqual(env["HIP_VISIBLE_DEVICES"], "0")
                    self.assertNotIn("ROCR_VISIBLE_DEVICES", env)
                    self.assertIn("BIGCHERRY_TEST_DETERMINISTIC_SEED", env)

    def test_subject_output_digest_difference_fails_bit_identical(self) -> None:
        result, _, _, _, _, _, _ = _run_producer(
            collect_kwargs={"output_mismatch": ("rd12_v_out", 2)},
        )
        correctness = result.correctness
        self.assertIsInstance(correctness, dict)
        self.assertEqual(correctness["disposition"], "failed")
        self.assertIn("lane='v'", str(correctness["detail"]))
        self.assertIn("seed=2", str(correctness["detail"]))
        self.assertIn("output_equal=False", str(correctness["detail"]))
        self.assertEqual(result.activation_evidence.status, "executed")

    def test_cpu_reference_digest_difference_fails_bit_identical(self) -> None:
        result, _, _, _, _, _, _ = _run_producer(
            collect_kwargs={"reference_mismatch": ("rd12_k_out", 1)},
        )
        correctness = result.correctness
        self.assertIsInstance(correctness, dict)
        self.assertEqual(correctness["disposition"], "failed")
        self.assertIn("reference_equal=False", str(correctness["detail"]))

    def test_output_element_count_difference_fails_bit_identical(self) -> None:
        result, _, _, _, _, _, _ = _run_producer(
            collect_kwargs={"nels_mismatch": ("rd12_v_out", 3)},
        )
        correctness = result.correctness
        self.assertIsInstance(correctness, dict)
        self.assertEqual(correctness["disposition"], "failed")
        self.assertIn("nels_equal=False", str(correctness["detail"]))

    def test_backend_reference_failure_is_audit_only(self) -> None:
        # RD12's correctness obligation is bit_identical: a
        # backend-reference threshold breach is persisted for audit but
        # does not flip the disposition (the historical producer
        # semantics, preserved by the mechanical migration).
        result, run_dir, _, _, _, _, _ = _run_producer(
            collect_kwargs={"backend_failure": ("rd12_k_out", 3)},
        )
        correctness = result.correctness
        self.assertIsInstance(correctness, dict)
        self.assertEqual(correctness["disposition"], "passed")
        doc = json.loads(
            (run_dir / "artifacts" / "rd12-correctness-gfx1100.json").read_text(
                encoding="utf-8"
            ),
        )
        self.assertFalse(doc["contract_results"]["backend_reference"]["passed"])
        self.assertIn(
            "seed=3",
            doc["contract_results"]["backend_reference"]["detail"],
        )

    def test_missing_subject_activation_marker_fails_closed(self) -> None:
        result, _, _, _, _, _, _ = _run_producer(subject_marker=False)
        correctness = result.correctness
        self.assertIsInstance(correctness, dict)
        self.assertEqual(correctness["disposition"], "failed")
        self.assertEqual(result.activation_evidence.status, "not_executed")
        self.assertIn(
            "focal dual-output MMVQ path was not proven active",
            str(correctness["detail"]),
        )

    def test_control_activation_marker_fails_closed(self) -> None:
        result, _, _, _, _, _, _ = _run_producer(control_marker=True)
        correctness = result.correctness
        self.assertIsInstance(correctness, dict)
        self.assertEqual(correctness["disposition"], "failed")
        self.assertEqual(result.activation_evidence.status, "not_executed")

    def test_per_arm_activation_logs_written_and_sha_bound(self) -> None:
        result, run_dir, _, _, _, _, _ = _run_producer()
        trace = result.trace_evidence
        self.assertIsInstance(trace, dict)

        positive = trace["positive"]
        self.assertIsInstance(positive, dict)
        negative = trace["negative"]
        self.assertIsInstance(negative, dict)
        positive_ref = positive["artifact"]
        self.assertIsInstance(positive_ref, dict)
        negative_ref = negative["artifact"]
        self.assertIsInstance(negative_ref, dict)
        self.assertEqual(
            positive_ref["path"],
            "artifacts/activation-rd12-gfx1100-subject.log",
        )
        self.assertEqual(
            negative_ref["path"],
            "artifacts/activation-rd12-gfx1100-control.log",
        )

        subject_text = (
            (run_dir / "artifacts" / "activation-rd12-gfx1100-subject.log")
            .read_bytes()
            .decode("utf-8")
        )
        control_text = (
            (run_dir / "artifacts" / "activation-rd12-gfx1100-control.log")
            .read_bytes()
            .decode("utf-8")
        )
        self.assertIn(TRACE_MARKER, subject_text)
        self.assertNotIn(TRACE_MARKER, control_text)
        # 6 invocations per arm, one header each.
        self.assertEqual(subject_text.count("\n---\n"), 5)
        self.assertEqual(control_text.count("\n---\n"), 5)
        for ref, text in ((positive_ref, subject_text), (negative_ref, control_text)):
            self.assertEqual(
                ref["sha256"],
                hashlib.sha256(text.encode("utf-8")).hexdigest(),
            )

    def test_logs_preserved_even_when_activation_fails(self) -> None:
        result, run_dir, _, _, _, _, _ = _run_producer(subject_marker=False)
        correctness = result.correctness
        self.assertIsInstance(correctness, dict)
        self.assertEqual(correctness["disposition"], "failed")
        trace = result.trace_evidence
        self.assertIsInstance(trace, dict)
        for role in ("positive", "negative"):
            observation = trace[role]
            self.assertIsInstance(observation, dict)
            artifact = observation["artifact"]
            self.assertIsInstance(artifact, dict)
            target = run_dir / str(artifact["path"])
            self.assertTrue(target.is_file())
            self.assertNotIn(
                TRACE_MARKER,
                target.read_bytes().decode("utf-8"),
            )

    def test_trace_evidence_satisfies_the_real_plan_validator(self) -> None:
        # The producer's artifact refs, bound by the SHARED binder from
        # the patch's own declared trace-marker check, must satisfy the
        # REAL validator (which re-reads the logs and re-verifies the
        # marker itself).
        from bigcherry.core import paths as bc_paths
        from bigcherry.patch import registry as patch_registry
        from bigcherry.patch import validation as patch_validation
        from bigcherry.patch import validation_campaign as vc
        from bigcherry.patch import validation_policy as patch_validation_policy

        result, run_dir, _, _, _, _, selection = _run_producer()

        registry = patch_registry.load_registry(bc_paths.PATCHES)
        descriptor = registry.get(SUBJECT_PATCH)
        plan = patch_validation_policy.require_execution_package(
            descriptor,
            root=bc_paths.PATCHES,
        )
        self.assertIsNotNone(plan)
        assert plan is not None
        bound = vc._bind_producer_trace_evidence(
            result.trace_evidence,
            validation_plan=plan,
            declared_artifacts=selection.spec.artifact_names,
            emitted_artifacts=result.emitted_artifacts,
        )
        marker_specs = [
            spec
            for spec in plan.checks
            if spec.capability == "activation" and spec.validator == "trace-marker"
        ]
        self.assertEqual(len(marker_specs), 1)
        spec = marker_specs[0]
        ctx = cast(
            "patch_validation.ValidationContext",
            SimpleNamespace(run_dir=run_dir, trace_evidence=bound),
        )
        verdict = patch_validation._builtin_trace_marker(spec, ctx)
        self.assertEqual(verdict.status, patch_validation.PASS, verdict.summary)

    def test_unsupported_or_multi_architecture_fails_before_build(self) -> None:
        for targets in (("gfx1151",), ("gfx1100", "gfx1201")):
            with self.subTest(targets=targets):
                temp = Path(tempfile.mkdtemp())
                run_dir = temp / "run"
                run_dir.mkdir()
                pair = _make_pair(temp)
                runtime = _FakeRuntime(run_dir=run_dir, pair=pair, device=None)
                ctx = vp.ProducerContext(
                    repo_root=TOOLS_ROOT.parent,
                    patch_dir=PATCH_DIR,
                    workdir=run_dir,
                    campaign_id=f"{SUBJECT_PATCH}/rd12",
                    base_revision="a" * 40,
                    hip_path=Path("/opt/rocm"),
                    fat_targets=vp.FatTargetPlan(targets=targets),
                    model=None,
                    corpus=None,
                    build_env={},
                    inputs={},
                    validation_build_identities={},
                    patch_id=SUBJECT_PATCH,
                    device_map={},
                    runtime=runtime,
                )
                selection = vp.resolve_producer(
                    patch_dir=PATCH_DIR,
                    producer_id="rd12",
                )
                with self.assertRaisesRegex(
                    vp.ValidationProducerError, "exactly one contract architecture"
                ):
                    selection.producer(ctx)
                # Fail fast: no build attempt.
                self.assertEqual(runtime.build_pair_calls, [])

    def test_no_device_for_run_architecture_fails_closed(self) -> None:
        with self.assertRaisesRegex(vp.ValidationProducerError, "no device mapped"):
            _run_producer(with_device=False)

    def test_malformed_backend_ops_evidence_propagates(self) -> None:
        with self.assertRaises(_EvidenceError):
            _run_producer(collect_kwargs={"malformed": ("rd12_k_out", 1)})


if __name__ == "__main__":
    unittest.main()
