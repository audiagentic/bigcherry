"""PA36 sub-slice 2 (T10, dev-gpt-agent req_2ecda033763949a9): the RD12
dedicated CLI path (--run-rd12-contract / _run_rd12_contract /
run_rd12_correctness_check) was DELETED from shared code and replaced by
the generic standard_campaign="run" producer path
(--validation-producer 1205_rd12_paired_mmvq_dual_output/rd12).

This file covers the replaced path at three levels:
  * source-layout regression guards (the deleted surface must not come
    back),
  * the shared evidence binder (binder-owned identity, plan-owned marker
    semantics, producer-supplied semantic measurements only),
  * the end-to-end dispatcher (_run_validation_producer) driven against
    the REAL 1205 patch/descriptor/plan with every expensive boundary
    (scaffold builds, binaries, subprocess) faked -- exit semantics,
    record identity facts, and the diagnostic-only skip path.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import inspect
import json
import os
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping, cast

TOOLS_ROOT = Path(__file__).resolve().parents[2]
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from bigcherry.patch import registry as patch_registry  # noqa: E402
from bigcherry.patch.campaign import producer as campaign_producer  # noqa: E402
from bigcherry.patch.campaign import scaffold as campaign_scaffold  # noqa: E402
from bigcherry.patch.campaign import build as campaign_build  # noqa: E402
from bigcherry.patch import evidence as patch_evidence  # noqa: E402
from bigcherry.patch import source as psi  # noqa: E402
from bigcherry.patch import validation_producer as vp  # noqa: E402
from bigcherry.patch.validation import ArtifactRef, ValidationContext, ValidationPlan  # noqa: E402
from bigcherry.tuning import correctness_evidence  # noqa: E402

PATCH_ID = "1205_rd12_paired_mmvq_dual_output"
CAMPAIGN_SRC = (
    TOOLS_ROOT / "bigcherry" / "patch" / "validation_campaign.py"
).read_text(encoding="utf-8")
TRACE_MARKER = "BIGCHERRY_PATCH_HIT patch=1205_rd12 path=dual_output_mmvq_fusion"


# ---------------------------------------------------------- layout guards


class RD12DedicatedPathDeletionTests(unittest.TestCase):
    def test_dedicated_cli_surface_is_gone(self) -> None:
        for needle in (
            "run_rd12_correctness_check",
            "_run_rd12_contract",
            "run_rd12_contract",
            "run-rd12-contract",
        ):
            self.assertNotIn(needle, CAMPAIGN_SRC, needle)

    def test_exclusion_tuples_do_not_carry_the_dead_flag(self) -> None:
        for line in CAMPAIGN_SRC.split("\n"):
            if '"run_rd58_state_restore", "run_rd73_contract"' in line:
                self.assertNotIn("run_rd12_contract", line)

    def test_generic_producer_path_is_the_only_rd12_entry(self) -> None:
        toml = TOOLS_ROOT.parent / "patches" / PATCH_ID / "validation" / "producer.toml"
        self.assertTrue(toml.is_file())
        self.assertIn('standard_campaign = "run"', toml.read_text(encoding="utf-8"))

    def test_contract_promotions_are_never_fed_dispositions(self) -> None:
        # req_f5ba56f4088e4742 finding #2: the promotion-semantic
        # persistence helpers (build_contract_evidence_for_persistence /
        # compute_persisted_validation_eligible) must never receive
        # execution.contract_verdicts -- producer check DISPOSITIONS are a
        # different semantic type than promotion-gate RESULTS. The
        # dispatcher persists producer_contract_promotions (built from the
        # producer's typed promotion_lane_effects; {} for non-promoting
        # producers, so every bound contract gets its explicit BLOCKED
        # ("no promotion result produced") verdict). (The skip-path
        # workdir diagnostic JSON may still DISPLAY
        # execution.contract_verdicts -- it never persists a record -- but
        # the two promotion-semantic PERSISTENCE helpers below must be
        # pinned to producer_contract_promotions, never
        # execution.contract_verdicts.)
        core_src = inspect.getsource(campaign_producer._run_validation_producer)
        self.assertIn(
            "build_contract_evidence_for_persistence(\n"
            "                validation_plan.contracts,\n"
            "                producer_contract_promotions,\n"
            "            )",
            core_src,
        )
        self.assertIn(
            "compute_persisted_validation_eligible(\n"
            "                descriptor,\n"
            "                execution.verdict,\n"
            "                producer_contract_promotions,",
            core_src,
        )
        # The promotion dict must be built from the producer's typed
        # promotion_lane_effects, never from execution.contract_verdicts
        # (dispositions are a different semantic type).
        self.assertIn(
            "execution.result.promotion_lane_effects.items()",
            core_src,
        )
        self.assertNotIn(
            "build_contract_evidence_for_persistence(\n"
            "                validation_plan.contracts,\n"
            "                execution.contract_verdicts",
            core_src,
        )

    def test_standard_campaign_scaffold_is_a_frozen_dataclass(self) -> None:
        # Real-hardware finding (sub-slice 3, 2026-09-17 gfx1030 run): the
        # committed class had LOST its @dataclass decorator, so its __init__
        # was the bare object one. Every offline test fakes
        # _build_standard_campaign_scaffold(), so the missing __init__ was
        # never executed -- the first real hardware run died with
        # "StandardCampaignScaffold() takes no arguments" AFTER all five
        # builds had completed. Pin the decorator contract structurally.
        self.assertTrue(dataclasses.is_dataclass(campaign_scaffold.StandardCampaignScaffold))
        # __dataclass_params__ is the only surface that exposes frozen=True;
        # it is private, so the attribute access needs a pyright ignore.
        self.assertTrue(
            campaign_scaffold.StandardCampaignScaffold.__dataclass_params__.frozen  # pyright: ignore[reportAttributeAccessIssue]
        )
        names = frozenset(
            f.name for f in dataclasses.fields(campaign_scaffold.StandardCampaignScaffold)
        )
        self.assertEqual(
            names,
            frozenset(
                {
                    "base_revision",
                    "control_composition",
                    "subject_composition",
                    "control_source",
                    "subject_source",
                    "stock_source",
                    "control_idempotent",
                    "subject_idempotent",
                    "build_root",
                    "build_env",
                    "tune_bin",
                    "replay_bin",
                    "stock_bin",
                    "control_bin",
                    "validation_subject_bin",
                    "tune_build_evidence",
                    "replay_build_evidence",
                    "stock_build_evidence",
                    "control_build_evidence",
                    "validation_subject_build_evidence",
                }
            ),
        )


# --------------------------------------------------------------- binders


def _binding_context(run_dir: Path) -> campaign_producer.ProducerEvidenceBindingContext:
    return campaign_producer.ProducerEvidenceBindingContext(
        run_dir=run_dir,
        patch_id=PATCH_ID,
        patch_path=(TOOLS_ROOT.parent / "patches" / PATCH_ID / "validation.toml"),
        base_revision="a" * 40,
        patched_source_tree="tree:subject",
        campaign_identity_digest="c" * 64,
        gpu_architectures=("gfx1100",),
    )


def _trace_plan(marker_specs: int = 1) -> ValidationPlan:
    specs = tuple(
        SimpleNamespace(
            capability="activation",
            validator="trace-marker",
            check_id=f"activation-{i}",
            config={"marker-regex": TRACE_MARKER},
        )
        for i in range(marker_specs)
    )
    return cast("ValidationPlan", SimpleNamespace(checks=specs))


class ProducerEvidenceBinderTests(unittest.TestCase):
    def test_bind_correctness_writes_the_root_document(self) -> None:
        run_dir = Path(tempfile.mkdtemp())
        binding = _binding_context(run_dir)
        document, bound = campaign_producer._bind_producer_correctness(
            {
                "disposition": "passed",
                "mechanism": "rd12-paired-mmvq-bit-identical",
                "detail": "all rows bit identical",
            },
            binding=binding,
        )
        assert document is not None, "binder must return the written document"
        self.assertEqual(document["disposition"], "passed")
        self.assertEqual(document["patch_id"], PATCH_ID)
        self.assertEqual(document["campaign_identity_digest"], "c" * 64)
        self.assertEqual(document["patched_source_tree"], "tree:subject")
        self.assertEqual(document["base_revision"], "a" * 40)

        path = run_dir / "correctness.json"
        self.assertTrue(path.is_file())
        on_disk = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(on_disk, document)
        self.assertEqual(set(bound), {"artifact"})
        artifact = cast(Mapping[str, object], bound["artifact"])
        self.assertIsInstance(artifact, Mapping)
        self.assertEqual(artifact["path"], "correctness.json")
        self.assertEqual(
            artifact["sha256"],
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )

    def test_bind_correctness_rejects_identity_smuggling(self) -> None:
        run_dir = Path(tempfile.mkdtemp())
        binding = _binding_context(run_dir)
        semantic = {
            "disposition": "passed",
            "mechanism": "m",
            "detail": "d",
            "campaign_identity_digest": "h" * 64,
        }
        with self.assertRaisesRegex(
            campaign_build.PatchCampaignError, "exactly " + r"{'disposition','mechanism','detail'}"
        ):
            campaign_producer._bind_producer_correctness(semantic, binding=binding)
        self.assertFalse((run_dir / "correctness.json").exists())

    def test_bind_correctness_rejects_bad_disposition(self) -> None:
        run_dir = Path(tempfile.mkdtemp())
        binding = _binding_context(run_dir)
        semantic = {
            "disposition": "maybe",
            "mechanism": "m",
            "detail": "d",
        }
        with self.assertRaisesRegex(campaign_build.PatchCampaignError, "disposition"):
            campaign_producer._bind_producer_correctness(semantic, binding=binding)

    def test_bind_correctness_none_passes_through(self) -> None:
        run_dir = Path(tempfile.mkdtemp())
        document, bound = campaign_producer._bind_producer_correctness(
            None,
            binding=_binding_context(run_dir),
        )
        self.assertIsNone(document)
        self.assertEqual(bound, {})
        self.assertFalse((run_dir / "correctness.json").exists())

    def test_bind_trace_injects_the_plan_owned_marker_regex(self) -> None:
        bound = campaign_producer._bind_producer_trace_evidence(
            {
                "positive": {"artifact": {"path": "artifacts/p.log", "sha256": "1"}},
                "negative": {"artifact": {"path": "artifacts/n.log", "sha256": "2"}},
            },
            validation_plan=_trace_plan(),
            declared_artifacts=frozenset({"p.log", "n.log"}),
            emitted_artifacts=frozenset({"p.log", "n.log"}),
        )
        self.assertEqual(set(bound), {"positive", "negative"})
        for role in ("positive", "negative"):
            observation = cast(Mapping[str, object], bound[role])
            self.assertIsInstance(observation, Mapping)
            self.assertEqual(observation["marker_regex"], TRACE_MARKER)
            self.assertIn("artifact", observation)

    def test_bind_trace_rejects_a_producer_owned_marker_regex(self) -> None:
        with self.assertRaisesRegex(campaign_build.PatchCampaignError, "exactly 'artifact'"):
            campaign_producer._bind_producer_trace_evidence(
                {
                    "positive": {
                        "artifact": {"path": "a", "sha256": "1"},
                        "marker_regex": ".*",
                    },
                    "negative": {"artifact": {"path": "b", "sha256": "2"}},
                },
                validation_plan=_trace_plan(),
                declared_artifacts=frozenset(),
                emitted_artifacts=frozenset(),
            )

    def _assert_one_trace_marker_required(self, marker_specs: int) -> None:
        trace = {
            "positive": {"artifact": {"path": "a", "sha256": "1"}},
            "negative": {"artifact": {"path": "b", "sha256": "2"}},
        }
        with self.assertRaisesRegex(campaign_build.PatchCampaignError, "exactly one trace-marker"):
            campaign_producer._bind_producer_trace_evidence(
                trace,
                validation_plan=_trace_plan(marker_specs),
                declared_artifacts=frozenset(),
                emitted_artifacts=frozenset(),
            )

    def test_bind_trace_zero_declared_checks_rejected(self) -> None:
        self._assert_one_trace_marker_required(0)

    def test_bind_trace_two_declared_checks_rejected(self) -> None:
        self._assert_one_trace_marker_required(2)

    def test_bind_result_evidence_orchestrates_run_path(self) -> None:
        from bigcherry.patch import activation as patch_activation

        run_dir = Path(tempfile.mkdtemp())
        binding = _binding_context(run_dir)
        plan = _trace_plan()
        ctx = ValidationContext(
            descriptor=cast("patch_registry.PatchDescriptor", object()),
            base_revision="a" * 40,
            control_source=None,
            subject_source=None,
        )
        result = vp.ProducerResult(
            correctness={
                "disposition": "passed",
                "mechanism": "m",
                "detail": "d",
            },
            validation_build_identities={
                "control": {"build_id": "c"},
                "subject": {"build_id": "s"},
            },
            activation_evidence=patch_activation.ActivationEvidence(
                status="executed",
                mechanism="rd12-trigger-marker",
                detail="ok",
            ),
            performance_evidence={},
            trace_evidence={
                "positive": {"artifact": {"path": "artifacts/p.log", "sha256": "1"}},
                "negative": {"artifact": {"path": "artifacts/n.log", "sha256": "2"}},
            },
            check_results=(),
            lane_effects=(),
            emitted_artifacts=frozenset({"p.log", "n.log"}),
        )
        bound = campaign_producer._bind_producer_result_evidence(
            result,
            validation_plan=plan,
            validation_context=ctx,
            binding=binding,
            declared_artifacts=frozenset({"p.log", "n.log"}),
        )
        # Trace evidence is the PLAN-BOUND form (marker_regex injected).
        trace = bound.validation_context.trace_evidence
        self.assertIsInstance(trace, Mapping)
        positive = trace["positive"]
        self.assertIsInstance(positive, Mapping)
        self.assertEqual(positive["marker_regex"], TRACE_MARKER)
        # Correctness evidence is the bound root artifact.
        correctness_ev = bound.validation_context.correctness_evidence
        self.assertIsInstance(correctness_ev, Mapping)
        artifact = correctness_ev["artifact"]
        self.assertIsInstance(artifact, Mapping)
        self.assertEqual(artifact["path"], "correctness.json")
        # Root artifacts are written under run_dir.
        self.assertTrue((run_dir / "correctness.json").is_file())
        activation_doc = json.loads(
            (run_dir / "activation.json").read_text(encoding="utf-8"),
        )
        self.assertEqual(activation_doc["campaign_identity_digest"], "c" * 64)
        # Disposition is computed once, by the shared verdict helper.
        self.assertEqual(bound.activation_disposition, activation_doc["verdict"])
        self.assertEqual(activation_doc["activation"]["status"], "executed")
        self.assertIsNotNone(bound.correctness)

    def test_bind_result_evidence_skip_path_passes_through(self) -> None:
        ctx = ValidationContext(
            descriptor=cast("patch_registry.PatchDescriptor", object()),
            base_revision="a" * 40,
            control_source=None,
            subject_source=None,
        )
        result = vp.ProducerResult(
            correctness=None,
            validation_build_identities={
                "control": {"build_id": "c"},
                "subject": {"build_id": "s"},
            },
            activation_evidence=None,
            performance_evidence=None,
            trace_evidence=None,
            check_results=(),
            lane_effects=(),
            emitted_artifacts=frozenset(),
        )
        bound = campaign_producer._bind_producer_result_evidence(
            result,
            validation_plan=_trace_plan(),
            validation_context=ctx,
            binding=None,
            declared_artifacts=frozenset(),
        )
        self.assertIsNone(bound.correctness)
        self.assertIsNone(bound.activation_disposition)
        self.assertEqual(bound.validation_context.trace_evidence, {})

    # ---- req_f5ba56f4088e4742 finding #1: manifest-ownership gate ------

    def test_bind_trace_rejects_undeclared_artifact_even_if_on_disk(self) -> None:
        # The exact bypass GPT named: a file that EXISTS under run_dir,
        # has the correct SHA, and is contained by run_dir -- but is NOT in
        # the producer manifest AND not claimed in emitted_artifacts --
        # must be rejected before any validator can see it.
        run_dir = Path(tempfile.mkdtemp())
        sneaky = run_dir / "artifacts" / "sneaky.log"
        sneaky.parent.mkdir(parents=True)
        sneaky.write_text("hit", encoding="utf-8")
        sha = hashlib.sha256(b"hit").hexdigest()
        with self.assertRaisesRegex(
            campaign_build.PatchCampaignError, "not declared in the producer manifest"
        ):
            campaign_producer._bind_producer_trace_evidence(
                {
                    "positive": {
                        "artifact": {"path": "artifacts/sneaky.log", "sha256": sha},
                    },
                    "negative": {
                        "artifact": {"path": "artifacts/n.log", "sha256": "2"}
                    },
                },
                validation_plan=_trace_plan(),
                declared_artifacts=frozenset({"n.log"}),
                emitted_artifacts=frozenset(),
            )

    def test_bind_trace_rejects_declared_but_not_emitted_artifact(self) -> None:
        # In the manifest but NOT claimed in this run's emitted_artifacts:
        # a producer may only bind what it actually wrote this run.
        with self.assertRaisesRegex(
            campaign_build.PatchCampaignError, "not claimed in result.emitted_artifacts"
        ):
            campaign_producer._bind_producer_trace_evidence(
                {
                    "positive": {
                        "artifact": {"path": "artifacts/p.log", "sha256": "1"}
                    },
                    "negative": {
                        "artifact": {"path": "artifacts/n.log", "sha256": "2"}
                    },
                },
                validation_plan=_trace_plan(),
                declared_artifacts=frozenset({"p.log", "n.log"}),
                emitted_artifacts=frozenset({"n.log"}),
            )

    def test_bind_trace_rejects_wrong_path_shape(self) -> None:
        # Not artifacts/<basename>: bare basename, nested, or a run-dir
        # escape must all fail before the filesystem is consulted.
        for bad in ("p.log", "artifacts/sub/p.log", "../outside.log"):
            with (
                self.subTest(path=bad),
                self.assertRaisesRegex(campaign_build.PatchCampaignError, "artifacts/<basename>"),
            ):
                campaign_producer._bind_producer_trace_evidence(
                    {
                        "positive": {"artifact": {"path": bad, "sha256": "1"}},
                        "negative": {
                            "artifact": {"path": "artifacts/n.log", "sha256": "2"}
                        },
                    },
                    validation_plan=_trace_plan(),
                    declared_artifacts=frozenset({"n.log"}),
                    emitted_artifacts=frozenset({"n.log"}),
                )

    def test_bind_result_rejects_undeclared_performance_artifact(self) -> None:
        # The same gate covers the generic performance_evidence["artifact"]
        # channel, not just trace evidence.
        ctx = ValidationContext(
            descriptor=cast("patch_registry.PatchDescriptor", object()),
            base_revision="a" * 40,
            control_source=None,
            subject_source=None,
        )
        result = vp.ProducerResult(
            correctness=None,
            validation_build_identities={
                "control": {"build_id": "c"},
                "subject": {"build_id": "s"},
            },
            activation_evidence=None,
            performance_evidence={
                "artifact": {"path": "artifacts/sneaky.json", "sha256": "1"},
            },
            trace_evidence=None,
            check_results=(),
            lane_effects=(),
            emitted_artifacts=frozenset(),
        )
        with self.assertRaisesRegex(
            campaign_build.PatchCampaignError, "not declared in the producer manifest"
        ):
            campaign_producer._bind_producer_result_evidence(
                result,
                validation_plan=_trace_plan(),
                validation_context=ctx,
                binding=None,
                declared_artifacts=frozenset(),
            )


# ------------------------------------------------------ dispatcher (end-to-end)


class _FakeBuildEvidence:
    def __init__(self, role: str) -> None:
        self.role = role

    @property
    def effective_build_id(self) -> str:
        return f"{self.role}-build-id"

    @property
    def effective_configure(self) -> dict[str, str]:
        return {"CMAKE_BUILD_TYPE": "Release", "GGML_HIP": "ON"}

    @property
    def verification(self) -> SimpleNamespace:
        return SimpleNamespace(to_dict=lambda: {"verified": True, "role": self.role})

    @property
    def runtime_artifacts(self) -> dict[str, object]:
        # Values must be 64-hex digests (evidence._validate_build_identity).
        digest = hashlib.sha256(self.role.encode("utf-8")).hexdigest()
        return {"main": digest}

    def campaign_identity(self) -> dict[str, object]:
        # The exact field set model_free_campaign_identity_digest()
        # validates (evidence._validate_build_identity).
        return {
            "effective_build_id": self.effective_build_id,
            "compile_verification_id": f"{self.role}-verify-id",
            "compile_commands_digest": f"{self.role}-cc-digest",
            "hip_compile_commands_digest": f"{self.role}-hip-cc-digest",
            "runtime_bundle_hash": f"{self.role}-bundle-hash",
            "runtime_artifacts": dict(sorted(self.runtime_artifacts.items())),
        }


class _FakeScaffold:
    def __init__(self, base_dir: Path) -> None:
        self.base_revision = "b" * 40
        self.control_composition = ((PATCH_ID, "c1"),)
        self.subject_composition = ((PATCH_ID, "c1"), (PATCH_ID, "c2"))
        self.control_source = base_dir / "trees" / "control"
        self.subject_source = base_dir / "trees" / "subject"
        self.stock_source = base_dir / "trees" / "stock"
        # The apply/build validators check the REAL directories exist.
        for source in (self.control_source, self.subject_source, self.stock_source):
            source.mkdir(parents=True, exist_ok=True)
        self.control_idempotent = True
        self.subject_idempotent = True
        # The real scaffold builds control/validation-subject with
        # llama-server+llama-bench targets under these bin dirs; the generic
        # path exposes them via ProducerContext.validation_binaries.
        self.control_bin = base_dir / "bin" / "control"
        self.validation_subject_bin = base_dir / "bin" / "validation-subject"
        self.tune_build_evidence = _FakeBuildEvidence("tune")
        self.replay_build_evidence = _FakeBuildEvidence("replay")
        self.stock_build_evidence = _FakeBuildEvidence("stock")
        self.control_build_evidence = _FakeBuildEvidence("control")
        self.validation_subject_build_evidence = _FakeBuildEvidence(
            "validation-subject",
        )

    @property
    def campaign_build_identities(self) -> dict[str, dict[str, object]]:
        return {
            role: getattr(self, f"{role}_build_evidence").campaign_identity()
            for role in ("tune", "replay", "stock")
        }

    @property
    def scaffold_validation_build_identities(self) -> dict[str, dict[str, object]]:
        return {
            "control": self.control_build_evidence.campaign_identity(),
            "subject": self.validation_subject_build_evidence.campaign_identity(),
        }


class _FakeDispatcherRuntime:
    """Stand-in for campaign_producer.CampaignProducerRuntime: one fake pair, real
    artifact files under run_dir, one device per run architecture."""

    def __init__(
        self,
        *,
        repo_root,
        patch_id,
        base_revision,
        workdir,
        hip_path,
        fat_targets,
        run_dir,
    ) -> None:
        self.run_dir = run_dir
        self.fat_targets = fat_targets
        self.pair = vp.ProducerBuildPair(
            base_revision=base_revision,
            control_source=workdir / "pair-trees" / "control",
            subject_source=workdir / "pair-trees" / "subject",
            control_composition=(
                ("1222_hi67_deterministic_test_backend_ops_seed", "p1"),
            ),
            subject_composition=(
                ("1222_hi67_deterministic_test_backend_ops_seed", "p1"),
                (PATCH_ID, "p2"),
            ),
            control_bin=workdir / "pair" / "CONTROL-BIN" / "test-backend-ops",
            subject_bin=workdir / "pair" / "SUBJECT-BIN" / "test-backend-ops",
            validation_build_identities={
                "control": {"build_id": "pair-control-build"},
                "subject": {"build_id": "pair-subject-build"},
            },
        )

    def build_pair(
        self,
        *,
        targets,
        primary_target,
        common_extra_patches=(),
        baseline_source="bigcherry",
        control_extra_cmake_args=(),
        subject_extra_cmake_args=(),
    ):
        return self.pair

    def device_contexts(self, *, device_map):
        from bigcherry.experiment.attestation import ExecutionIdentity

        architecture = self.fat_targets.targets[0]
        return (
            vp.ProducerDeviceContext(
                architecture=architecture,
                device_index=0,
                execution_identity=ExecutionIdentity(
                    backend="ROCm",
                    architectures=(architecture,),
                ),
                env_overrides={"HIP_VISIBLE_DEVICES": "0"},
                env_unset=("ROCR_VISIBLE_DEVICES",),
            ),
        )

    def write_artifact(self, *, name, payload):
        return self._write(name, json.dumps(payload, indent=2))

    def write_text_artifact(self, *, name, text):
        return self._write(name, text)

    def _write(self, name: str, text: str) -> ArtifactRef:
        path = self.run_dir / "artifacts" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("utf-8"))
        return ArtifactRef(
            name=name,
            path=path.relative_to(self.run_dir).as_posix(),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )


def _fake_tree(path: Path) -> str:
    """Real psi.git_worktree_tree() returns a git tree hash (hex); the
    campaign identity validator requires 40/64 hex for the source trees."""
    return hashlib.sha256(str(path).encode("utf-8")).hexdigest()


def _fake_collect_pass():
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
        assert seed is not None
        assert runner is not None
        runner(
            [str(binary)],
            capture_output=True,
            text=True,
            env={"BIGCHERRY_TEST_DETERMINISTIC_SEED": str(seed)},
        )
        return SimpleNamespace(
            seed=seed,
            reference_digest=f"input-{seed}",
            e_n_nmse=1e-6,
            max_abs_native=0.25,
            threshold_t=1e-3,
            native_execution_status="ok",
            native_output_digest=f"gpu-{target_tensor}-{seed}",
            reference_output_digest=f"cpu-{target_tensor}-{seed}",
            output_nels=1024,
        )

    return collect_native_seed_evidence


def _fake_collect_subject_mismatch():
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
        assert seed is not None
        assert runner is not None
        runner(
            [str(binary)],
            capture_output=True,
            text=True,
            env={"BIGCHERRY_TEST_DETERMINISTIC_SEED": str(seed)},
        )
        native = f"gpu-{target_tensor}-{seed}"
        if "SUBJECT-BIN" in str(binary) and target_tensor == "rd12_k_out" and seed == 1:
            native += "-mismatch"
        return SimpleNamespace(
            seed=seed,
            reference_digest=f"input-{seed}",
            e_n_nmse=1e-6,
            max_abs_native=0.25,
            threshold_t=1e-3,
            native_execution_status="ok",
            native_output_digest=native,
            reference_output_digest=f"cpu-{target_tensor}-{seed}",
            output_nels=1024,
        )

    return collect_native_seed_evidence


def _fake_subprocess_pass():
    def run(argv, **kwargs):
        executable = str(argv[0])
        emit = "SUBJECT-BIN" in executable
        return SimpleNamespace(
            returncode=0,
            stdout="",
            stderr=(TRACE_MARKER + "\n" if emit else ""),
        )

    return run


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs) -> dict[str, object]:
        self.calls.append(kwargs)
        return {"record": f"sentinel-{len(self.calls) - 1}"}


def _args(tmp: Path, **overrides) -> argparse.Namespace:
    base = {
        "patch": PATCH_ID,
        "model": None,
        "hip_path": Path("/opt/rocm"),
        "amdgpu_targets": "gfx1100",
        "device_map": ["gfx1100=0"],
        "workdir": tmp / "workdir",
        "worktree_root": tmp / "worktrees",
        "build_root": tmp / "build",
        "correctness_evidence": None,
        "run_performance_benchmark": False,
        "producer_corpus": None,
        "baseline_source": "bigcherry",
        "bench_prompt": 512,
        "bench_gen": 128,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def _dispatch(
    tmp: Path,
    *,
    collect=None,
    selection=None,
    args_overrides: dict[str, object] | None = None,
    recorders: dict[str, _Recorder] | None = None,
):
    """Run campaign_producer._run_validation_producer() against the REAL 1205 patch with
    every expensive boundary faked. Returns a namespace of everything the
    tests assert on."""
    collect = collect or _fake_collect_pass()
    fake_run = _fake_subprocess_pass()
    recorders = recorders or {}
    scaffold_recorder = recorders.setdefault("scaffold", _Recorder())
    make_record_recorder = recorders.setdefault("make_record", _Recorder())
    write_record_recorder = recorders.setdefault("write_record", _Recorder())

    scaffold = _FakeScaffold(tmp / "scaffold")
    env_backup = {key: os.environ.get(key) for key in ("ROCM_PATH", "HIP_PATH", "PATH")}

    def fake_scaffold(**kwargs) -> _FakeScaffold:
        scaffold_recorder.calls.append(kwargs)
        return scaffold

    def fake_write_record(record) -> Path:
        write_record_recorder.calls.append({"record": record})
        path = tmp / "records" / f"record-{len(write_record_recorder.calls) - 1}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"persisted": True}), encoding="utf-8")
        return path

    patches = [
        mock.patch.object(campaign_producer, "_build_standard_campaign_scaffold", fake_scaffold),
        mock.patch.object(campaign_producer, "CampaignProducerRuntime", _FakeDispatcherRuntime),
        mock.patch.object(psi, "git_worktree_tree", _fake_tree),
        mock.patch.object(psi, "patch_implementation_digest", lambda pid: "d" * 64),
        mock.patch.object(psi, "composition_digest", lambda c: "e" * 64),
        mock.patch.object(
            correctness_evidence, "collect_native_seed_evidence", collect
        ),
        mock.patch("subprocess.run", fake_run),
        mock.patch.object(patch_evidence, "make_record", make_record_recorder),
        mock.patch.object(patch_evidence, "write_record", fake_write_record),
    ]
    if selection is not None:
        patches.append(
            mock.patch.object(campaign_producer, "resolve_producer", lambda **kw: selection)
        )
    for patcher in patches:
        patcher.start()
    try:
        exit_code = campaign_producer._run_validation_producer(
            _args(tmp, **(args_overrides or {})),
            producer_id="rd12",
            provided_inputs={},
        )
    finally:
        for key, value in env_backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        for patcher in patches:
            patcher.stop()

    run_dir = tmp / "workdir" / "campaign"
    outcome_path = run_dir / "producer-execution.json"
    outcome = (
        json.loads(outcome_path.read_text(encoding="utf-8"))
        if outcome_path.is_file()
        else None
    )
    return SimpleNamespace(
        exit_code=exit_code,
        scaffold=scaffold,
        scaffold_calls=scaffold_recorder.calls,
        make_record_calls=make_record_recorder.calls,
        write_record_calls=write_record_recorder.calls,
        run_dir=run_dir,
        outcome=outcome,
    )


class RD12GenericDispatcherTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = Path(tempfile.mkdtemp())

    def tearDown(self) -> None:
        import shutil

        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_run_path_exit_zero_and_record_persisted(self) -> None:
        result = _dispatch(self._tmp)

        # Exit 0 iff execution + binding + persistence completed --
        # eligibility is evidence, not process success.
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(len(result.make_record_calls), 1)
        self.assertEqual(len(result.write_record_calls), 1)
        self.assertEqual(len(result.scaffold_calls), 1)
        scaffold_args = result.scaffold_calls[0]
        self.assertEqual(scaffold_args["patch_id"], PATCH_ID)
        self.assertEqual(scaffold_args["baseline_source"], "bigcherry")
        self.assertEqual(scaffold_args["amdgpu_targets"], "gfx1100")

        self.assertIsNotNone(result.outcome)
        assert result.outcome is not None
        # apply/build/activation/correctness PASS; performance/controls
        # stay BLOCKED (the producer declares both benchmark CLIs
        # forbidden) -- so the verdict is ineligible but the run succeeds.
        self.assertFalse(result.outcome["eligible"])
        # The tracked record WAS persisted (exit 0 requires it).
        self.assertIsInstance(result.outcome["evidence_record"], str)
        check_results = result.outcome["check_results"]
        # Scaffold-backed and producer-backed checks PASS.
        self.assertEqual(check_results["apply"]["status"], "pass")
        self.assertEqual(check_results["build"]["status"], "pass")
        self.assertEqual(check_results["activation"]["status"], "pass")
        self.assertEqual(check_results["correctness"]["status"], "pass")
        # The producer forbids both benchmark CLIs: performance/controls
        # can never pass from this path -- the verdict stays ineligible.
        for check_id in ("performance", "controls"):
            self.assertNotEqual(check_results[check_id]["status"], "pass")

        # Root canonical evidence written by the binder.
        correctness = json.loads(
            (self._tmp / "workdir" / "campaign" / "correctness.json").read_text(
                encoding="utf-8"
            ),
        )
        self.assertEqual(correctness["disposition"], "passed")
        self.assertTrue(
            (self._tmp / "workdir" / "campaign" / "activation.json").is_file(),
        )

    def test_run_path_record_identity_facts(self) -> None:
        result = _dispatch(self._tmp)
        self.assertEqual(len(result.make_record_calls), 1)
        kwargs = result.make_record_calls[0]

        self.assertEqual(kwargs["patch_id"], PATCH_ID)
        self.assertEqual(kwargs["base_ref"], cfg_pinned())
        self.assertEqual(kwargs["base_revision"], "b" * 40)
        self.assertEqual(
            kwargs["patched_source_tree"], _fake_tree(result.scaffold.subject_source)
        )
        self.assertEqual(kwargs["gpu_architectures"], "gfx1100")
        self.assertEqual(kwargs["campaign_workdir"], result.run_dir)
        # Two DISTINCT provenance domains: campaign identities from the
        # scaffold, validation identities from the producer's own pair.
        self.assertEqual(
            kwargs["build_identities"],
            result.scaffold.campaign_build_identities,
        )
        self.assertEqual(
            kwargs["validation_build_identities"],
            {
                "control": {"build_id": "pair-control-build"},
                "subject": {"build_id": "pair-subject-build"},
            },
        )
        # The model-free campaign identity is computed from the SCAFFOLD
        # facts (never the fat-three).
        expected_digest = patch_evidence.model_free_campaign_identity_digest(
            patch_name=PATCH_ID,
            patch_digest="d" * 64,
            patched_source_tree=_fake_tree(result.scaffold.subject_source),
            gpu_architecture="gfx1100",
            campaign_build_identities=result.scaffold.campaign_build_identities,
            base_revision=result.scaffold.base_revision,
        )
        self.assertEqual(kwargs["campaign_identity_digest"], expected_digest)
        # Producer-owned artifact allowlist, bound correctness, activation.
        self.assertEqual(kwargs["producer_artifact_names"], _spec_artifact_names())
        bound_correctness = kwargs["correctness"]
        self.assertIsInstance(bound_correctness, Mapping)
        self.assertEqual(bound_correctness["disposition"], "passed")
        self.assertIsNotNone(kwargs["activation_disposition"])
        # Lane effects from the producer result.
        self.assertEqual(kwargs["lane_effects"], ())

    def test_run_path_subject_mismatch_still_exit_zero(self) -> None:
        result = _dispatch(self._tmp, collect=_fake_collect_subject_mismatch())
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(len(result.make_record_calls), 1)
        kwargs = result.make_record_calls[0]
        bound_correctness = kwargs["correctness"]
        self.assertIsInstance(bound_correctness, Mapping)
        self.assertEqual(bound_correctness["disposition"], "failed")
        self.assertFalse(result.outcome["eligible"])
        check_results = result.outcome["check_results"]
        self.assertEqual(check_results["correctness"]["status"], "fail")
        # The record is still persisted -- an INELIGIBLE record is the
        # whole point of tracked evidence.
        self.assertFalse(kwargs["validation_eligible"])

    def test_forbid_cli_gates_fail_before_the_scaffold(self) -> None:
        for overrides, pattern in (
            (
                {"correctness_evidence": self._tmp / "ce.json"},
                "--correctness-evidence is ambiguous",
            ),
            (
                {"run_performance_benchmark": True},
                "--run-performance-benchmark is mutually exclusive",
            ),
        ):
            with (
                self.subTest(overrides=sorted(overrides)),
                self.assertRaisesRegex(vp.ValidationProducerError, pattern),
            ):
                _dispatch(
                    self._tmp,
                    args_overrides=cast("dict[str, object]", overrides),
                )

    def test_producer_patch_id_mismatch_fails_closed(self) -> None:
        selection = vp.ProducerSelection(
            spec=vp.ProducerSpec(
                patch_id="9999_not_this_patch",
                producer_id="rd12",
                entrypoint=Path("producer.py"),
                callable_name="run",
                trace_probe="skip",
                standard_campaign="run",
                correctness_evidence_cli="forbid",
                performance_benchmark_cli="forbid",
                artifact_names=frozenset(),
            ),
            producer=lambda ctx: None,  # type: ignore[return-value]
        )
        with self.assertRaisesRegex(vp.ValidationProducerError, "does not match"):
            _dispatch(self._tmp, selection=selection)

    def test_skip_path_no_scaffold_no_record(self) -> None:
        def trivial_producer(ctx):
            return vp.ProducerResult(
                correctness=None,
                validation_build_identities={
                    "control": {"build_id": "c"},
                    "subject": {"build_id": "s"},
                },
                activation_evidence=None,
                performance_evidence=None,
                trace_evidence=None,
                check_results=(),
                lane_effects=(),
                emitted_artifacts=frozenset(),
            )

        selection = vp.ProducerSelection(
            spec=vp.ProducerSpec(
                patch_id=PATCH_ID,
                producer_id="rd12",
                entrypoint=Path("producer.py"),
                callable_name="run",
                trace_probe="skip",
                standard_campaign="skip",
                correctness_evidence_cli="forbid",
                performance_benchmark_cli="forbid",
                artifact_names=frozenset(),
            ),
            producer=trivial_producer,
        )
        result = _dispatch(self._tmp, selection=selection)

        self.assertEqual(result.exit_code, 0)
        self.assertEqual(result.scaffold_calls, [])
        self.assertEqual(result.make_record_calls, [])
        self.assertEqual(result.write_record_calls, [])
        # Self-contained producer semantics: its own run_dir, diagnostic
        # outcome only, no tracked record.
        skip_run_dir = self._tmp / "workdir" / "producer" / "rd12"
        self.assertTrue((skip_run_dir / "producer-execution.json").is_file())
        self.assertIsNone(result.outcome)  # campaign/ outcome never written


def cfg_pinned() -> str:
    from bigcherry.core import config as campaign_config
    from bigcherry.core import paths as bc_paths

    return campaign_config.load(bc_paths.RECIPES).pinned


def _spec_artifact_names() -> frozenset[str]:
    selection = vp.resolve_producer(
        patch_dir=TOOLS_ROOT.parent / "patches" / PATCH_ID,
        producer_id="rd12",
    )
    return selection.spec.artifact_names


if __name__ == "__main__":
    unittest.main()
