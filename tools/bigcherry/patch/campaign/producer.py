"""Validation-producer execution runtime for the patch-validation campaign
(distinct from bigcherry.patch.validation_producer, which owns the
producer spec/result contract)."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import sys
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from bigcherry.build.builds import capture_completed_build_evidence
from bigcherry.core.paths import REPO_ROOT
from bigcherry.experiment import contract as experiment_contract
from bigcherry.experiment.attestation import ExecutionIdentity
from bigcherry.patch.activation import (
    ActivationEvidence,
    verdict,
    write_activation_json,
)
from bigcherry.patch.campaign.benchmark import (
    parse_device_map,
    run_paired_llama_benchmark,
)
from bigcherry.patch.campaign.build import (
    _atomic_write_json,
    _full_requested_cmake_args,
    _hip_env,
    _hip_only,
    _print,
    _ROCR_VISIBLE_DEVICES_UNSET,
    _write_bound_artifact,
    build_tree,
    LLAMA_CPP_SRC,
    PatchCampaignError,
)
from bigcherry.patch.campaign.contract import (
    assert_validation_subject_parity,
    build_contract_evidence_for_persistence,
    compute_contract_correctness_gate,
    compute_persisted_validation_eligible,
)
from bigcherry.patch.campaign.scaffold import (
    _build_standard_campaign_scaffold,
    StandardCampaignScaffold,
)
from bigcherry.patch.campaign.trace import (
    _run_one_trace_probe,
    run_trace_activation_probes,
)
from bigcherry.patch.validation import (
    ArtifactRef,
    ValidationContext,
    ValidationPlan,
    ValidationResult,
    Verdict,
)
from bigcherry.patch.validation_producer import (
    FatTargetPlan,
    ProducerBuildPair,
    ProducerContext,
    ProducerDeviceContext,
    ProducerPairedBenchmarkOutcome,
    ProducerResult,
    ProducerSelection,
    resolve_producer,
    validate_producer_cli_compatibility,
    validate_producer_inputs,
    validate_producer_result,
    ValidationProducerError,
)


# --------------------------------------------------------- PA36-F producer runtime


@dataclass(frozen=True)
class CampaignProducerRuntime:
    """The one concrete ``ProducerRuntime`` implementation (PA36-F step 1,
    GPT design req_8ec9b90c05f84a30). Patch-local producer modules never
    import this module directly -- they receive an instance of this class
    through ``ProducerContext.runtime`` instead, which is the sanctioned
    dependency-direction seam validation_producer.py's module docstring
    establishes.

    Every method here is a thin, faithful wrapper around the existing
    generic primitives already living in this module
    (``build_tree()``/``capture_completed_build_evidence()``/
    ``resolve_selected_device_execution_identity()``/
    ``run_paired_llama_benchmark()``/``source.resolve_source_composition()``
    /``source.materialize_composition()``) -- nothing here reimplements
    them."""

    repo_root: Path
    patch_id: str
    base_revision: str
    workdir: Path
    hip_path: Path
    fat_targets: FatTargetPlan
    run_dir: Path

    def build_pair(
        self,
        *,
        targets: tuple[str, ...],
        primary_target: str,
        common_extra_patches: tuple[str, ...] = (),
        baseline_source: str = "bigcherry",
        control_extra_cmake_args: tuple[str, ...] = (),
        subject_extra_cmake_args: tuple[str, ...] = (),
        require_parity: bool = False,
    ) -> ProducerBuildPair:
        """The one authority for the PA36 build-once-fat-multiarch rule:
        exactly one control build and one subject build, both at the same
        target set, regardless of how many devices or architectures the
        producer will later run against.

        ``targets`` is the authoritative, REQUIRED target set (validated
        through FatTargetPlan, so its non-empty/no-duplicate invariants hold;
        a producer may request a fat gfx1100;gfx1201;gfx1030 set while the
        outer run names one execution arch). ``common_extra_patches``
        resolves into BOTH arms (control = baseline + common; subject =
        baseline + common + focal) so a producer whose correctness pair
        needs supplementary evidence patches in both arms passes them here
        instead of materializing its own pair below the build authority
        (dev-gpt-agent req_98777b7a51f84820 + review req_052817cb66d14bc1)."""
        # targets and primary_target are REQUIRED (GPT review
        # req_052817cb66d14bc1): the pre-existing API already required both,
        # so silent defaults would only add a path where a future producer
        # accidentally builds the wrong fat plan or the wrong binary. The
        # producer names the explicit target set (typically
        # ctx.fat_targets.targets); FatTargetPlan validates it, and its
        # cmake_value is the exact AMDGPU_TARGETS string. The binary build
        # list still comes from primary_target -- build_tree() itself only
        # ever builds the single requested target for a producer build pair
        # (a producer wanting multiple binaries calls build_pair() once per
        # binary set, never widens this one call).
        target_plan = FatTargetPlan(targets=targets)
        from bigcherry.patch import source as psi

        control_revision, control_composition = psi.resolve_source_composition(
            baseline_source,
            focal=None,
            extra_patches=common_extra_patches,
            base_ref=self.base_revision,
            base_repo=LLAMA_CPP_SRC,
        )
        subject_revision, subject_composition = psi.resolve_source_composition(
            baseline_source,
            focal=self.patch_id,
            extra_patches=common_extra_patches,
            base_ref=self.base_revision,
            base_repo=LLAMA_CPP_SRC,
        )
        if control_revision != subject_revision:
            raise PatchCampaignError(
                f"{self.patch_id}: build_pair() control/subject resolved different "
                f"base revisions ({control_revision!r} vs {subject_revision!r})"
            )

        control_src = psi.materialize_composition(
            base_repo=LLAMA_CPP_SRC,
            worktree_root=self.workdir / "control",
            resolved_revision=control_revision,
            composition=control_composition,
            overlay_root=psi.REPO_ROOT / "src",
            requested_revision=self.base_revision,
        )
        subject_src = psi.materialize_composition(
            base_repo=LLAMA_CPP_SRC,
            worktree_root=self.workdir / "subject",
            resolved_revision=subject_revision,
            composition=subject_composition,
            overlay_root=psi.REPO_ROOT / "src",
            requested_revision=self.base_revision,
        )

        exe = ".exe" if sys.platform == "win32" else ""
        build_root = self.workdir / "builds"
        # Directory NAME must not contain ';' -- target_plan.cmake_value is the
        # correct CMake AMDGPU_TARGETS *value* (semicolon-joined, as CMake
        # list syntax requires), but a build directory literally named with
        # embedded semicolons breaks CMake's own internal argument handling
        # (real failure found on real hardware, PA39 real-hardware acceptance
        # attempt #3b: "execute_process given unknown argument 'gfx1201'"
        # during compiler-id detection, because CMake treats ';' in certain
        # internal strings as its own list separator). Use a '+'-joined slug
        # for the directory name only; the actual cmake invocation still
        # receives the real semicolon-joined value.
        target_slug = "+".join(target_plan.targets)
        control_name = f"{self.patch_id}-control-{target_slug}"
        subject_name = f"{self.patch_id}-subject-{target_slug}"

        control_bin = build_tree(
            name=control_name,
            hip_path=self.hip_path,
            amdgpu_targets=target_plan.cmake_value,
            workdir=build_root,
            targets=[primary_target],
            source=control_src,
            extra_cmake_args=list(control_extra_cmake_args),
        )
        subject_bin = build_tree(
            name=subject_name,
            hip_path=self.hip_path,
            amdgpu_targets=target_plan.cmake_value,
            workdir=build_root,
            targets=[primary_target],
            source=subject_src,
            extra_cmake_args=list(subject_extra_cmake_args),
        )

        build_env = _hip_env(self.hip_path)
        control_binary = control_bin / f"{primary_target}{exe}"
        subject_binary = subject_bin / f"{primary_target}{exe}"
        control_cmake_args = _full_requested_cmake_args(
            hip_path=self.hip_path,
            amdgpu_targets=target_plan.cmake_value,
            extra_cmake_args=list(control_extra_cmake_args),
        )
        subject_cmake_args = _full_requested_cmake_args(
            hip_path=self.hip_path,
            amdgpu_targets=target_plan.cmake_value,
            extra_cmake_args=list(subject_extra_cmake_args),
        )
        control_build_evidence = capture_completed_build_evidence(
            build_root / control_name,
            source_root=control_src,
            architecture=target_plan.targets,
            binary=control_binary,
            requested_cmake_args=control_cmake_args,
            build_env=build_env,
        )
        subject_build_evidence = capture_completed_build_evidence(
            build_root / subject_name,
            source_root=subject_src,
            architecture=target_plan.targets,
            binary=subject_binary,
            requested_cmake_args=subject_cmake_args,
            build_env=build_env,
        )

        if require_parity:
            assert_validation_subject_parity(
                control_build_evidence,
                subject_build_evidence,
                patch_id=self.patch_id,
            )
        return ProducerBuildPair(
            base_revision=control_revision,
            control_source=control_src,
            subject_source=subject_src,
            control_composition=tuple(control_composition),
            subject_composition=tuple(subject_composition),
            control_bin=control_binary,
            subject_bin=subject_binary,
            validation_build_identities={
                "control": control_build_evidence.campaign_identity(),
                "subject": subject_build_evidence.campaign_identity(),
            },
        )

    def device_contexts(
        self,
        *,
        device_map: Mapping[str, tuple[int, ...]],
    ) -> tuple[ProducerDeviceContext, ...]:
        """The only producer-facing device selector (PA36-F step 1):
        reuses ``resolve_selected_device_execution_identity()``'s real
        device-inventory verification per explicit index rather than
        mutating ambient HIP_VISIBLE_DEVICES, and always returns the
        HIP-only env shape (``env_overrides={"HIP_VISIBLE_DEVICES":
        str(index)}``, ``env_unset=("ROCR_VISIBLE_DEVICES",)``) -- no
        ambient-only selector and no ROCR/HIP double-filtering (PNRO17)."""
        contexts: list[ProducerDeviceContext] = []
        for architecture, indices in device_map.items():
            for index in indices:
                identity, _selector_env, locator = self._resolve_one_device(
                    architecture,
                    index,
                )
                contexts.append(
                    ProducerDeviceContext(
                        architecture=architecture,
                        device_index=index,
                        execution_identity=identity,
                        env_overrides={"HIP_VISIBLE_DEVICES": str(index)},
                        env_unset=_ROCR_VISIBLE_DEVICES_UNSET,
                        locator=locator,
                    )
                )
        return tuple(contexts)

    def _resolve_one_device(
        self,
        architecture: str,
        index: int,
    ) -> tuple[ExecutionIdentity, dict[str, str], str]:
        """Resolve exactly ONE device index against the real host
        inventory -- the same fail-closed checks
        ``resolve_selected_device_execution_identity()`` performs from
        ``HIP_VISIBLE_DEVICES``, applied here to an explicit index
        instead of reading ambient environment, so a multi-device
        producer never has to mutate process-global env to select each
        device in turn."""
        from bigcherry.core import environment as bc_environment

        host_devices = bc_environment.load_default().host().devices
        matches = [d for d in host_devices if d.index == index]
        if not matches:
            raise PatchCampaignError(
                f"{self.patch_id}: device_contexts() index {index} is not a "
                f"configured device in config/environment.toml (known indices: "
                f"{sorted(d.index for d in host_devices)})"
            )
        device = matches[0]
        if device.locator is None:
            raise PatchCampaignError(
                f"{self.patch_id}: device_contexts() index {index} "
                f"({device.arch}) has no verified locator in config/environment.toml"
            )
        if device.arch != architecture:
            raise PatchCampaignError(
                f"{self.patch_id}: device_contexts() index {index} is configured "
                f"as {device.arch!r}, but {architecture!r} was requested"
            )
        # NOT locators=(device.locator,): the observed side of this
        # comparison comes from parse_rocm_attestation() parsing a plain
        # llama-bench/llama-perplexity stdout device banner ("Device 0:
        # AMD Radeon Graphics, gfx1201 (0x1201), ..."), which has no PCI
        # locator at all and hardcodes ObservedDevice.locator=None --
        # structurally, not as a bug in that parser (llama.cpp does not
        # print one). Passing a real verified locator here made
        # compare_execution_identity() require a match this observation
        # channel can never supply, so every real-hardware run through
        # this generic --validation-producer device_contexts() path was
        # guaranteed to fail closed at the first paired-benchmark call
        # (found on real hardware, PA39 real-hardware acceptance attempt
        # #3c). The two other ExecutionIdentity(...) call sites in this
        # same file (~4931, ~5952) already omit locators for the exact
        # same reason -- this now matches that established precedent
        # rather than inventing a new, weaker guarantee: device identity
        # here rests on architecture match + the HIP_VISIBLE_DEVICES
        # env-scoping this method already returns below, same as those
        # sibling call sites. Known residual gap (documented, not fixed
        # here, matching this item's existing gfx1100-coverage-gap
        # documentation pattern for RD06/RD07): for RD07's gfx1100 arm,
        # which has TWO real physical devices (index 0 and 1, both
        # architecture gfx1100), this evidence channel cannot distinguish
        # which of the two actually ran -- that relies on
        # HIP_VISIBLE_DEVICES scoping alone, unverified in the
        # attestation record, for gfx1100 specifically. RD05/RD06 use
        # gfx1201 only, the sole device of that architecture on this
        # host, so the gap does not apply to them.
        identity = ExecutionIdentity(
            backend="ROCm",
            architectures=(device.arch,),
        )
        return identity, {"HIP_VISIBLE_DEVICES": str(index)}, device.locator

    def write_artifact(self, *, name: str, payload: JsonObject):
        return _write_bound_artifact_ref(self.run_dir, name, payload)

    def write_text_artifact(self, *, name: str, text: str) -> ArtifactRef:
        return _write_bound_text_artifact_ref(self.run_dir, name, text)

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
        device: ProducerDeviceContext | None = None,
        env_overrides: Mapping[str, str] | None = None,
        env_unset: tuple[str, ...] = (),
    ) -> ProducerPairedBenchmarkOutcome:
        # RD58 (PA36 migration #4, dev-gpt-agent
        # req_ecb4b77a4c4e4bdd MAJOR #2): the env selector authority
        # is the DEVICE, not the caller. When device != None, reject
        # selector keys (HIP_VISIBLE_DEVICES / ROCR_VISIBLE_DEVICES)
        # in the caller's env_overrides/env_unset (the caller could
        # otherwise replace the device selector or delete it, while
        # execution_identity still describes the device object), and
        # apply the device selector LAST (so it wins over any
        # non-selector collision). When device is None (RD58
        # multi-GPU), preserve the explicit ambient
        # HIP_VISIBLE_DEVICES (reject it in caller env_overrides so
        # it is not replaced); the caller may set non-selector
        # overrides (e.g. GGML_CUDA_REGISTER_HOST=1) and unset
        # ROCR_VISIBLE_DEVICES.
        _SELECTOR_KEYS = frozenset({"HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"})
        if device is not None:
            for _key in env_overrides or {}:
                if _key in _SELECTOR_KEYS:
                    raise PatchCampaignError(
                        f"run_paired_llama_benchmark: caller "
                        f"env_overrides must not set selector key "
                        f"{_key!r} when a device context is supplied "
                        "(the device owns the selector)"
                    )
            for _key in env_unset:
                if _key in _SELECTOR_KEYS:
                    raise PatchCampaignError(
                        f"run_paired_llama_benchmark: caller "
                        f"env_unset must not unset selector key "
                        f"{_key!r} when a device context is supplied "
                        "(the device owns the selector)"
                    )
            # Caller's non-selector overrides first, device selector
            # LAST (device wins on any collision).
            merged_overrides: dict[str, str] = {}
            if env_overrides:
                merged_overrides.update(env_overrides)
            merged_overrides.update(_hip_only(dict(device.env_overrides)))
        else:
            # device=None (RD58 multi-GPU): preserve the explicit
            # ambient HIP_VISIBLE_DEVICES (reject it in caller
            # env_overrides so it is not replaced); the caller may
            # set non-selector overrides + unset ROCR_VISIBLE_DEVICES.
            for _key in env_overrides or {}:
                if _key == "HIP_VISIBLE_DEVICES":
                    raise PatchCampaignError(
                        "run_paired_llama_benchmark: caller "
                        "env_overrides must not set "
                        "HIP_VISIBLE_DEVICES when device is None "
                        "(the explicit ambient selector is preserved)"
                    )
            # RD58 (PA36 migration #4, dev-gpt-agent
            # req_2c5e7a0230914eab MAJOR): also reject
            # HIP_VISIBLE_DEVICES in env_unset -- a multi-GPU
            # producer could otherwise delete the authoritative
            # ambient selector via env_unset=("HIP_VISIBLE_DEVICES",
            # ), defeating the preflight and running with
            # unrestricted visibility. ROCR_VISIBLE_DEVICES remains
            # allowed/expected in env_unset.
            for _key in env_unset:
                if _key == "HIP_VISIBLE_DEVICES":
                    raise PatchCampaignError(
                        "run_paired_llama_benchmark: caller "
                        "env_unset must not unset "
                        "HIP_VISIBLE_DEVICES when device is None "
                        "(the explicit ambient selector is preserved)"
                    )
            merged_overrides = dict(env_overrides) if env_overrides else {}
        execution_identity = device.execution_identity if device is not None else None
        outcome = run_paired_llama_benchmark(
            control_binary=control_binary,
            subject_binary=subject_binary,
            model=model,
            hip_path=self.hip_path,
            workloads=workloads,
            patch_args=patch_args,
            runtime_args=runtime_args,
            pairs=pairs,
            log_context=log_context,
            env_overrides=merged_overrides or None,
            env_unset=env_unset,
            execution_identity=execution_identity,
        )
        return ProducerPairedBenchmarkOutcome(
            runs=outcome.runs,
            commands=outcome.commands,
            raw_logs=tuple(outcome.raw_logs),
        )

    def build_materialized_pair(
        self,
        *,
        control_source: Path,
        subject_source: Path,
        targets: tuple[str, ...],
        primary_target: str,
    ) -> ProducerBuildPair:
        """Build and capture identities for already-materialized
        control/subject sources (RD08 sub-slice 3, GPT
        req_c2e69928e8b34de0). The sources are already materialized
        by the producer (e.g. rd08_correctness.materialize_rd08_
        variants()); this method only builds the binaries and
        captures the build identities."""
        from bigcherry.build import builds as builds_module

        target_plan = FatTargetPlan(targets=targets)
        exe = ".exe" if sys.platform == "win32" else ""
        build_root = self.workdir / "builds"

        target_slug = "+".join(target_plan.targets)
        control_name = f"{self.patch_id}-materialized-control-{target_slug}"
        subject_name = f"{self.patch_id}-materialized-subject-{target_slug}"

        build_env = _hip_env(self.hip_path)
        control_bin = build_tree(
            name=control_name,
            hip_path=self.hip_path,
            amdgpu_targets=target_plan.cmake_value,
            workdir=build_root,
            targets=[primary_target],
            source=control_source,
            extra_cmake_args=[],
        )
        subject_bin = build_tree(
            name=subject_name,
            hip_path=self.hip_path,
            amdgpu_targets=target_plan.cmake_value,
            workdir=build_root,
            targets=[primary_target],
            source=subject_source,
            extra_cmake_args=[],
        )
        control_binary = control_bin / f"{primary_target}{exe}"
        subject_binary = subject_bin / f"{primary_target}{exe}"

        cmake_args = _full_requested_cmake_args(
            hip_path=self.hip_path,
            amdgpu_targets=target_plan.cmake_value,
            extra_cmake_args=[],
        )
        control_evidence = builds_module.capture_completed_build_evidence(
            build_root / control_name,
            source_root=control_source,
            architecture=target_plan.targets,
            binary=control_binary,
            requested_cmake_args=cmake_args,
            build_env=build_env,
        )
        subject_evidence = builds_module.capture_completed_build_evidence(
            build_root / subject_name,
            source_root=subject_source,
            architecture=target_plan.targets,
            binary=subject_binary,
            requested_cmake_args=cmake_args,
            build_env=build_env,
        )

        # GPT review: do not fabricate compositions from tree SHA
        control_composition = ()
        subject_composition = ()

        return ProducerBuildPair(
            base_revision=self.base_revision,
            control_source=control_source,
            subject_source=subject_source,
            control_composition=control_composition,
            subject_composition=subject_composition,
            control_bin=control_binary,
            subject_bin=subject_binary,
            validation_build_identities={
                "control": control_evidence.campaign_identity(),
                "subject": subject_evidence.campaign_identity(),
            },
        )

    def run_trace_probe(
        self,
        *,
        binary: Path,
        model: Path,
        device: ProducerDeviceContext,
        bench_prompt: int,
        bench_gen: int,
        log_context: str,
        disable_fusion: bool = False,
    ) -> str:
        """Run one trace probe (RD08 sub-slice 3, GPT
        req_c2e69928e8b34de0). Wraps the existing _run_one_trace_probe()
        primitive -- does not create a second implementation."""
        return _run_one_trace_probe(
            name=log_context,
            binary=binary,
            model=model,
            hip_path=self.hip_path,
            workdir=self.workdir,
            bench_prompt=bench_prompt,
            bench_gen=bench_gen,
            disable_fusion=disable_fusion,
            env_overrides=dict(device.env_overrides) if device.env_overrides else None,
            env_unset=tuple(device.env_unset) if device.env_unset else (),
        )


def _write_bound_artifact_ref(run_dir: Path, name: str, payload: JsonObject):
    """``_write_bound_artifact()`` returns a plain ``{"path", "sha256"}``
    dict; ``ProducerRuntime.write_artifact()`` must return the real typed
    ``ArtifactRef`` a ``ValidationResult.artifacts``/``validate_producer_
    result()`` can bind -- this wraps the former into the latter without
    duplicating the write logic."""
    ref = _write_bound_artifact(run_dir, name, payload)
    return ArtifactRef(name=name, path=ref["path"], sha256=ref["sha256"])


def _write_bound_text_artifact_ref(run_dir: Path, name: str, text: str) -> ArtifactRef:
    """``write_artifact()`` serializes a JSON payload; ``write_text_artifact()``
    writes raw text VERBATIM (RD12's raw per-arm activation logs are not JSON,
    and re-serializing them would change the exact bytes the trace-marker
    validator re-reads). Same bound-artifact contract as write_artifact():
    returns the typed ArtifactRef a ValidationResult.artifacts /
    validate_producer_result() can bind."""
    target = run_dir / "artifacts" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    # Byte-verbatim: a text-mode write would translate newlines on Windows
    # (\n -> \r\n) and change the exact bytes (and sha256) the trace-marker
    # validator re-reads (GPT review req_052817cb66d14bc1).
    data = text.encode("utf-8")
    target.write_bytes(data)
    return ArtifactRef(
        name=name,
        path=target.relative_to(run_dir).as_posix(),
        sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
    )


@dataclass(frozen=True)
class ProducerEvidenceBindingContext:
    """The shared-code identity facts a standard_campaign="run" producer's
    semantic evidence gets bound into (PA36 sub-slice 2, dev-gpt-agent
    req_2ecda033763949a9 T3). The producer supplies ONLY semantic
    measurements; the binder owns every canonical identity field -- patch
    identity, source tree, campaign identity digest -- and writes the
    root-level evidence artifacts under run_dir."""

    run_dir: Path
    patch_id: str
    patch_path: Path
    base_revision: str
    patched_source_tree: str
    campaign_identity_digest: str
    gpu_architectures: tuple[str, ...]


@dataclass(frozen=True)
class BoundProducerEvidence:
    """The evaluation context AFTER a producer's evidence was bound (T3):
    the same ValidationContext shape with trace/correctness/performance
    evidence replaced by their canonical bound forms, plus the root
    correctness document and activation disposition the executor must
    carry into make_record()."""

    validation_context: ValidationContext
    correctness: dict[str, object] | None
    activation_disposition: str | None


def _bind_producer_correctness(
    semantic: JsonObject | None,
    *,
    binding: ProducerEvidenceBindingContext,
) -> tuple[dict[str, object] | None, dict[str, object]]:
    """Bind a producer's semantic correctness measurement into the
    canonical correctness document (T3). The producer supplies EXACTLY
    {disposition, mechanism, detail} -- no identity fields, which the
    shared binder owns: the root correctness.json under run_dir, its
    artifact binding, and the schema/patch/source/campaign identity."""
    from bigcherry.patch import evidence as patch_validation_evidence

    if semantic is None:
        return None, {}

    values = dict(semantic)
    expected = {"disposition", "mechanism", "detail"}
    if set(values) != expected:
        raise PatchCampaignError(
            "producer correctness must contain exactly "
            "{'disposition','mechanism','detail'}"
        )

    disposition = values["disposition"]
    mechanism = values["mechanism"]
    detail = values["detail"]
    if disposition not in ("passed", "failed"):
        raise PatchCampaignError(
            f"producer correctness disposition must be passed/failed, got {disposition!r}"
        )
    if not isinstance(mechanism, str) or not mechanism:
        raise PatchCampaignError("producer correctness mechanism must be non-empty")
    if not isinstance(detail, str):
        raise PatchCampaignError("producer correctness detail must be a string")

    document = {
        "schema_version": patch_validation_evidence.CORRECTNESS_SCHEMA_VERSION,
        "patch_id": binding.patch_id,
        "patch_validation_subject_digest": patch_validation_evidence.patch_validation_subject_digest(
            binding.patch_path
        ),
        "base_revision": binding.base_revision,
        "patched_source_tree": binding.patched_source_tree,
        "campaign_identity_digest": binding.campaign_identity_digest,
        "gpu_architectures": list(binding.gpu_architectures),
        "disposition": disposition,
        "mechanism": mechanism,
        "detail": detail,
    }

    path = binding.run_dir / "correctness.json"
    _atomic_write_json(path, document)
    artifact = {
        "path": path.relative_to(binding.run_dir).as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    return document, {"artifact": artifact}


def _producer_owned_artifact_ref(
    artifact: object,
    role: str,
    *,
    declared: frozenset[str],
    emitted: frozenset[str],
) -> dict[str, object]:
    """Manifest-ownership gate on producer-supplied artifact refs
    (dev-gpt-agent req_f5ba56f4088e4742 finding #1): the ref must be
    exactly {path, sha256}, the path exactly 'artifacts/<basename>', and
    the basename must appear BOTH in the producer's declared manifest
    (spec.artifact_names) and in result.emitted_artifacts.
    _artifact_is_bound() alone proves only path containment + SHA --
    without this gate a producer could write an undeclared file and
    smuggle it into the persisted record through trace/performance
    evidence."""
    if not isinstance(artifact, Mapping):
        raise PatchCampaignError(
            f"producer {role} artifact reference must be an object"
        )
    if set(artifact) != {"path", "sha256"}:
        raise PatchCampaignError(
            f"producer {role} artifact reference must contain exactly "
            f"path/sha256, got {sorted(artifact)!r}"
        )
    path = artifact["path"]
    sha256 = artifact["sha256"]
    if not isinstance(path, str) or not isinstance(sha256, str):
        raise PatchCampaignError(
            f"producer {role} artifact reference path/sha256 must be strings"
        )
    if "\\" in path:
        raise PatchCampaignError(
            f"producer {role} artifact path must use forward slashes: {path!r}"
        )
    parts = path.split("/")
    if len(parts) != 2 or parts[0] != "artifacts" or not parts[1]:
        raise PatchCampaignError(
            f"producer {role} artifact path must be exactly "
            f"'artifacts/<basename>', got {path!r}"
        )
    basename = parts[1]
    if basename not in declared:
        raise PatchCampaignError(
            f"producer {role} artifact {basename!r} is not declared in the "
            f"producer manifest; declared: {sorted(declared)!r}"
        )
    if basename not in emitted:
        raise PatchCampaignError(
            f"producer {role} artifact {basename!r} was not claimed in "
            f"result.emitted_artifacts"
        )
    return {"path": path, "sha256": sha256}


def _bind_producer_trace_evidence(
    trace_evidence: JsonObject | None,
    *,
    validation_plan: ValidationPlan,
    declared_artifacts: frozenset[str],
    emitted_artifacts: frozenset[str],
) -> dict[str, object]:
    """Bind a producer's trace observations into the canonical
    trace_evidence shape (T3). The producer owns ONLY the positive/negative
    artifact refs; the plan owns the marker semantics -- marker_regex is
    injected from the single declared trace-marker activation check, and a
    producer-supplied regex is rejected. Every artifact ref is additionally
    manifest-gated (declared AND emitted) before evaluation."""
    if trace_evidence is None:
        return {}

    trace_specs = tuple(
        spec
        for spec in validation_plan.checks
        if spec.capability == "activation" and spec.validator == "trace-marker"
    )
    if len(trace_specs) != 1:
        raise PatchCampaignError(
            "producer supplied trace evidence but validation plan does not "
            "declare exactly one trace-marker activation check"
        )

    marker = trace_specs[0].config.get("marker-regex")
    if not isinstance(marker, str) or not marker:
        raise PatchCampaignError(
            "trace-marker activation check has no non-empty marker-regex"
        )

    raw = dict(trace_evidence)
    if set(raw) != {"positive", "negative"}:
        raise PatchCampaignError(
            "producer trace evidence must contain exactly positive/negative"
        )

    bound: dict[str, object] = {}
    for role in ("positive", "negative"):
        observation = raw[role]
        if not isinstance(observation, Mapping):
            raise PatchCampaignError(
                f"producer trace evidence {role} must be an object"
            )
        if set(observation) != {"artifact"}:
            raise PatchCampaignError(
                f"producer trace evidence {role} must contain exactly 'artifact'"
            )
        bound[role] = {
            "marker_regex": marker,
            "artifact": _producer_owned_artifact_ref(
                observation["artifact"],
                f"trace {role}",
                declared=declared_artifacts,
                emitted=emitted_artifacts,
            ),
        }
    return bound


def _bind_producer_result_evidence(
    result: ProducerResult,
    *,
    validation_plan: ValidationPlan,
    validation_context: ValidationContext,
    binding: ProducerEvidenceBindingContext | None,
    declared_artifacts: frozenset[str],
) -> BoundProducerEvidence:
    """One generic post-producer evidence-binding pass (T3/T4): trace,
    performance, correctness, and activation evidence are replaced by
    their canonical bound forms so the fallback validators -- and
    compute_verdict() -- see exactly what gets persisted. When ``binding``
    is None (self-contained "skip" producers) the context passes through
    unchanged except for producer-supplied evidence dicts. Every
    producer-supplied artifact ref is manifest-gated against
    ``declared_artifacts`` AND ``result.emitted_artifacts``
    (req_f5ba56f4088e4742 finding #1)."""
    trace_evidence = (
        _bind_producer_trace_evidence(
            result.trace_evidence,
            validation_plan=validation_plan,
            declared_artifacts=declared_artifacts,
            emitted_artifacts=result.emitted_artifacts,
        )
        if result.trace_evidence is not None
        else validation_context.trace_evidence
    )
    performance_evidence = (
        dict(result.performance_evidence)
        if result.performance_evidence is not None
        else validation_context.performance_evidence
    )
    if "artifact" in performance_evidence:
        performance_evidence = {
            **performance_evidence,
            "artifact": _producer_owned_artifact_ref(
                performance_evidence["artifact"],
                "performance",
                declared=declared_artifacts,
                emitted=result.emitted_artifacts,
            ),
        }

    correctness: dict[str, object] | None = None
    correctness_evidence = validation_context.correctness_evidence
    activation_disposition: str | None = None

    if binding is not None:
        correctness, correctness_evidence = _bind_producer_correctness(
            result.correctness,
            binding=binding,
        )

        activation_evidence = result.activation_evidence
        if activation_evidence is not None:
            if not isinstance(activation_evidence, ActivationEvidence):
                raise PatchCampaignError(
                    "ProducerResult.activation_evidence must be ActivationEvidence or None"
                )
            activation_disposition = verdict(
                activation_evidence,
                correctness_passed=None,
            )
            write_activation_json(
                binding.run_dir / "activation.json",
                activation_evidence,
                activation_disposition,
                extra={
                    "campaign_identity_digest": binding.campaign_identity_digest,
                },
            )

    return BoundProducerEvidence(
        validation_context=dataclasses.replace(
            validation_context,
            trace_evidence=trace_evidence,
            correctness_evidence=correctness_evidence,
            performance_evidence=performance_evidence,
        ),
        correctness=correctness,
        activation_disposition=activation_disposition,
    )


@dataclass(frozen=True)
class ProducerExecution:
    """The full result of running one producer through the generic
    dispatcher (PA36-F step 3): the typed producer result plus every
    downstream value ``make_record()`` needs, computed exactly once so
    the caller never has to re-derive them. ``bound_correctness`` and
    ``activation_disposition`` are the T3/T4 bound evidence values the
    caller persists -- never re-derived."""

    selection: ProducerSelection
    result: ProducerResult
    evaluated: Mapping[str, ValidationResult]
    verdict: Verdict
    contract_verdicts: Mapping[str, JsonObject]
    bound_correctness: JsonObject | None = None
    activation_disposition: str | None = None
    # trace_probe="run": the dispatcher's own scaffold-probe activation
    # evidence (the producer returns none in that mode).
    activation_evidence: ActivationEvidence | None = None


def _run_producer_trace_probes(
    *,
    result: ProducerResult,
    producer_context: ProducerContext,
    validation_plan: ValidationPlan,
    run_dir: Path,
    bench_prompt: int,
    bench_gen: int,
) -> tuple[ActivationEvidence, dict[str, object]]:
    """trace_probe="run" (dev-gpt-agent req_110d0729beb44d8b Q3): the
    dispatcher itself runs the scaffold's generic two-probe activation
    probe against the standard-campaign scaffold's SUBJECT llama-bench --
    the producer returns NO activation/trace evidence in this mode (both
    are rejected fail-closed), and this probe's ActivationEvidence plus
    its bound log refs are the record's canonical activation evidence.
    The marker comes from the single declared trace-marker check; the
    description uses the standard campaign's fallback
    (``<patch_id> activation``)."""
    trace_specs = tuple(
        spec
        for spec in validation_plan.checks
        if spec.capability == "activation" and spec.validator == "trace-marker"
    )
    if len(trace_specs) != 1:
        raise PatchCampaignError(
            "trace_probe='run' requires the validation plan to declare "
            "exactly one trace-marker activation check"
        )
    marker = trace_specs[0].config.get("marker-regex")
    if not isinstance(marker, str) or not marker:
        raise PatchCampaignError(
            "trace_probe='run': trace-marker activation check has no "
            "non-empty marker-regex"
        )
    if result.activation_evidence is not None or result.trace_evidence is not None:
        raise PatchCampaignError(
            "trace_probe='run' producers must not supply their own "
            "activation/trace evidence -- the dispatcher runs the "
            "scaffold's generic probe"
        )
    if producer_context.model is None:
        raise PatchCampaignError(
            "trace_probe='run' requires a model (the probe is a real llama-bench run)"
        )
    subject_binaries = producer_context.validation_binaries.get("subject")
    if (
        not isinstance(subject_binaries, Mapping)
        or "llama-bench" not in subject_binaries
    ):
        raise PatchCampaignError(
            "trace_probe='run' requires the standard-campaign scaffold "
            "subject llama-bench binary"
        )
    # PA36 RD13/1206 migration (GPT req_760c0fe82d7b4609 MAJOR): the
    # activation probe must run on the SAME physical device as the
    # correctness measurement -- resolve exactly one device for the
    # invocation architecture and thread its HIP-only selector env
    # (never ambient visibility).
    run_architecture = producer_context.fat_targets.targets[0]
    devices = producer_context.runtime.device_contexts(
        device_map=producer_context.device_map,
    )
    device_matches = [d for d in devices if d.architecture == run_architecture]
    if len(device_matches) != 1:
        raise PatchCampaignError(
            f"trace_probe='run': expected exactly one device mapped for "
            f"run architecture {run_architecture!r}, got {len(device_matches)}; "
            "--device-map must select one real device for it"
        )
    probe_device = device_matches[0]
    probe = run_trace_activation_probes(
        marker_regex=marker,
        description=f"{producer_context.patch_id} activation",
        binary=subject_binaries["llama-bench"],
        model=producer_context.model,
        hip_path=producer_context.hip_path,
        workdir=run_dir,
        bench_prompt=bench_prompt,
        bench_gen=bench_gen,
        env_overrides=dict(probe_device.env_overrides),
        env_unset=probe_device.env_unset,
    )
    if probe is None:
        raise PatchCampaignError(
            "trace_probe='run': the trace probe unexpectedly returned None "
            "(marker_regex and description were both non-empty)"
        )
    return probe


def execute_validation_producer(
    *,
    patch_dir: Path,
    producer_id: str,
    provided_inputs: Mapping[str, str],
    producer_context: ProducerContext,
    validation_plan: ValidationPlan,
    validation_context: ValidationContext,
    correctness_evidence_requested: bool,
    performance_benchmark_requested: bool,
    selection: ProducerSelection | None = None,
    evidence_binding_context: ProducerEvidenceBindingContext | None = None,
    bench_prompt: int = 512,
    bench_gen: int = 128,
) -> ProducerExecution:
    """The one generic entry point that executes a selected patch-local
    validation producer end to end (PA36-F step 3, GPT design
    req_8ec9b90c05f84a30). ``validation_campaign.py`` knows how to run
    the producer this resolves; it never knows which patch/RD it is --
    all patch identity lives in ``patch_dir``/``producer_id`` and the
    already-constructed ``producer_context``.

    Exact sequence (per GPT's design, section 3):
    1. resolve_producer()
    2. assert selection.spec.patch_id == producer_context.patch_id
    3. validate_producer_inputs()
    4. validate_producer_cli_compatibility()
    5. dataclasses.replace(producer_context, inputs=validated_inputs)
    6. result = selection.producer(context) -- any exception fails closed
       as PatchCampaignError; a partial/half-built result is never used.
    7. validate_producer_result()
    8. build producer_results = {check_id: validation_result}
    9. for every plan check in order: producer-supplied result, else
       evaluate_check()
    10. verdict = compute_verdict()
    11. contract_verdicts from every check_result carrying a disposition
    12. return ProducerExecution(...)
    """
    from bigcherry.patch import validation as patch_validation

    selection = selection or resolve_producer(
        patch_dir=patch_dir,
        producer_id=producer_id,
    )

    if selection.spec.patch_id != producer_context.patch_id:
        raise ValidationProducerError(
            f"execute_validation_producer: resolved producer patch_id "
            f"{selection.spec.patch_id!r} does not match "
            f"producer_context.patch_id {producer_context.patch_id!r}"
        )

    validated_inputs = validate_producer_inputs(selection.spec, provided_inputs)
    validate_producer_cli_compatibility(
        selection.spec,
        correctness_evidence_requested=correctness_evidence_requested,
        performance_benchmark_requested=performance_benchmark_requested,
    )

    context = dataclasses.replace(producer_context, inputs=validated_inputs)

    try:
        result = selection.producer(context)
    except Exception as exc:
        raise PatchCampaignError(
            f"{selection.spec.patch_id}/{selection.spec.producer_id}: producer raised: {exc}"
        ) from exc
    if not isinstance(result, ProducerResult):
        raise PatchCampaignError(
            f"{selection.spec.patch_id}/{selection.spec.producer_id}: producer returned "
            f"{type(result).__name__}, not ProducerResult"
        )

    validate_producer_result(
        selection.spec,
        result,
        plan=validation_plan,
        context=validation_context,
    )

    # trace_probe="run" (dev-gpt-agent req_110d0729beb44d8b Q3): the
    # dispatcher itself runs the scaffold's generic two-probe activation
    # probe; the producer supplies no activation/trace evidence in that
    # mode. The probe's ActivationEvidence + bound log refs replace the
    # context's trace_evidence BEFORE the T3/T4 binding pass, so the
    # trace-marker fallback validator evaluates the real probe output.
    probe_evidence: ActivationEvidence | None = None
    probe_disposition: str | None = None
    if selection.spec.trace_probe == "run":
        if evidence_binding_context is None:
            raise PatchCampaignError(
                "trace_probe='run' requires a standard campaign (the "
                "scaffold subject llama-bench + a bound run_dir)"
            )
        probe_evidence, probe_detail = _run_producer_trace_probes(
            result=result,
            producer_context=context,
            validation_plan=validation_plan,
            run_dir=evidence_binding_context.run_dir,
            bench_prompt=bench_prompt,
            bench_gen=bench_gen,
        )
        from bigcherry.patch.activation import (
            verdict as _activation_verdict,
        )

        probe_disposition = _activation_verdict(probe_evidence, correctness_passed=None)
        write_activation_json(
            evidence_binding_context.run_dir / "activation.json",
            probe_evidence,
            probe_disposition,
            extra={
                "campaign_identity_digest": evidence_binding_context.campaign_identity_digest,
                "trace_probe": probe_detail,
            },
        )
        _print(f"activation: {probe_evidence.status} ({probe_evidence.mechanism})")

        def _probe_log(
            detail: dict[str, object],
            role: str,
            key: str,
        ) -> str:
            observation = detail[role]
            if not isinstance(observation, Mapping):
                raise PatchCampaignError(
                    f"trace probe detail {role!r} must be an object"
                )
            value = observation[key]
            if not isinstance(value, str):
                raise PatchCampaignError(
                    f"trace probe detail {role}.{key} must be a string"
                )
            return value

        def _bind_producer_log(relative_log_path: str) -> dict[str, str]:
            target = (evidence_binding_context.run_dir / relative_log_path).resolve()
            return {
                "path": relative_log_path,
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            }

        validation_context = dataclasses.replace(
            validation_context,
            trace_evidence={
                "positive": {
                    "marker_regex": probe_detail["marker_regex"],
                    "artifact": _bind_producer_log(
                        _probe_log(probe_detail, "positive", "log")
                    ),
                },
                "negative": {
                    "marker_regex": probe_detail["marker_regex"],
                    "artifact": _bind_producer_log(
                        _probe_log(probe_detail, "negative_control", "log")
                    ),
                },
            },
        )

    # T3/T4 (dev-gpt-agent req_2ecda033763949a9): generic post-producer
    # evidence binding -- the fallback validators see the canonical bound
    # evidence, not the raw producer dicts, before compute_verdict().
    bound = _bind_producer_result_evidence(
        result,
        validation_plan=validation_plan,
        validation_context=validation_context,
        binding=evidence_binding_context,
        declared_artifacts=selection.spec.artifact_names,
    )

    producer_results: dict[str, ValidationResult] = {
        record.check_id: record.validation_result for record in result.check_results
    }
    evaluated: dict[str, ValidationResult] = {}
    for spec in validation_plan.checks:
        if spec.check_id in producer_results:
            evaluated[spec.check_id] = producer_results[spec.check_id]
        else:
            evaluated[spec.check_id] = patch_validation.evaluate_check(
                spec,
                bound.validation_context,
            )

    verdict = patch_validation.compute_verdict(validation_plan, evaluated)

    contract_verdicts: dict[str, JsonObject] = {}
    for record in result.check_results:
        if record.disposition is not None:
            (contract_id,) = record.contract_ids
            contract_verdicts[contract_id] = record.disposition

    return ProducerExecution(
        selection=selection,
        result=result,
        evaluated=evaluated,
        verdict=verdict,
        contract_verdicts=contract_verdicts,
        bound_correctness=bound.correctness,
        activation_disposition=(
            bound.activation_disposition
            if bound.activation_disposition is not None
            else probe_disposition
        ),
        activation_evidence=probe_evidence,
    )


# --------------------------------------------------------- PA36-F step 5: CLI


def _parse_validation_producer_selector(value: str) -> tuple[str, str]:
    """``PATCH/PRODUCER_ID`` -> ``(patch_id, producer_id)``. Uses
    ``rsplit("/", 1)`` (per GPT design section 5) so a producer_id can never
    itself be misread as containing a ``/`` -- only the patch component
    could (it never does today, but the split direction is the one that
    stays correct if it ever did)."""
    if "/" not in value:
        raise PatchCampaignError(
            f"--validation-producer {value!r} must be PATCH/PRODUCER_ID"
        )
    patch_id, producer_id = value.rsplit("/", 1)
    if not patch_id or not producer_id:
        raise PatchCampaignError(
            f"--validation-producer {value!r}: patch and producer id must both be non-empty"
        )
    return patch_id, producer_id


def _parse_producer_inputs(values: Iterable[str]) -> dict[str, str]:
    """Repeated ``--producer-input NAME=VALUE`` -> ``{name: value}``. Fails
    closed on a missing ``=``, an empty name, or the same name given
    twice -- ambiguity is never silently resolved by last-one-wins."""
    inputs: dict[str, str] = {}
    for raw in values:
        if "=" not in raw:
            raise PatchCampaignError(f"--producer-input {raw!r} must be NAME=VALUE")
        name, _, value = raw.partition("=")
        if not name:
            raise PatchCampaignError(
                f"--producer-input {raw!r}: name must not be empty"
            )
        if name in inputs:
            raise PatchCampaignError(f"--producer-input: duplicate name {name!r}")
        inputs[name] = value
    return inputs


def _parse_producer_device_map(entries: list[str]) -> dict[str, tuple[int, ...]]:
    """Same ARCH=ID[,ID...] shape as ``parse_device_map()``, but a
    producer's ``ProducerRuntime.device_contexts()`` addresses real devices
    by integer index (``bigcherry.core.environment`` device inventory), not
    the opaque string ids the legacy ``--run-performance-benchmark`` device
    pool uses -- so this fails closed on a non-integer id instead of
    passing an unusable string through."""
    raw_map = parse_device_map(entries)
    device_map: dict[str, tuple[int, ...]] = {}
    for architecture, ids in raw_map.items():
        try:
            device_map[architecture] = tuple(int(i) for i in ids)
        except ValueError as exc:
            raise PatchCampaignError(
                f"--device-map {architecture}={','.join(ids)}: device ids must be integers "
                "for --validation-producer"
            ) from exc
    return device_map


def _producer_file_identity(path: Path) -> dict[str, object]:
    """GPT review req_7a72896b609a48b5 BLOCKER #1: the real file facts
    ({path, size, sha256}) of a producer input, bound into the campaign
    identity. A producer run that depends on model/corpus BYTES must show
    that dependency in its identity -- two runs with different model or
    corpus bytes never share a campaign identity. Fails closed when the
    file is missing."""
    if not path.is_file():
        raise PatchCampaignError(f"producer input file missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path),
        "size": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


@dataclass(frozen=True)
class _ProducerCampaignSetup:
    """Run-directory, validation context and scaffold bindings for one producer run."""

    scaffold: StandardCampaignScaffold | None
    evidence_binding: ProducerEvidenceBindingContext | None
    campaign_identity_digest: str | None
    base_revision: str
    run_dir: Path
    validation_context: object
    scaffold_validation_ids: Mapping
    scaffold_validation_binaries: Mapping


def _validate_producer_architectures(
    args: argparse.Namespace,
    bound_contracts,
    fat_targets: FatTargetPlan,
    device_map,
) -> None:
    # PA36 (dev-gpt-agent req_19c0ea3d2d3f40db): validate requested AND
    # measured architectures against the union of bound contracts'
    # scope.architectures before producer execution. This is the generic,
    # contract-authority guard -- not a per-patch hardcoded check.
    if fat_targets.targets or device_map:
        contract_architectures: set[str] = set()
        for _contract in bound_contracts:
            if _contract.scope.architectures:
                contract_architectures.update(_contract.scope.architectures)
        if contract_architectures:
            # Validate requested (fat) architectures
            if fat_targets.targets:
                requested_archs = set(fat_targets.targets)
                unsupported_requested = requested_archs - contract_architectures
                if unsupported_requested:
                    raise PatchCampaignError(
                        f"{args.patch}: requested architectures "
                        f"{sorted(unsupported_requested)} are not in the "
                        f"bound contract scope {sorted(contract_architectures)}"
                    )
            # Validate measured (device_map) architectures
            if device_map:
                measured_archs = set(device_map.keys())
                unsupported_measured = measured_archs - contract_architectures
                if unsupported_measured:
                    raise PatchCampaignError(
                        f"{args.patch}: measured architectures "
                        f"{sorted(unsupported_measured)} are not in the "
                        f"bound contract scope {sorted(contract_architectures)}"
                    )
                # Require measured architectures to be a subset of fat targets
                # so a selected device cannot run against a binary not built
                # for that architecture
                if fat_targets.targets:
                    not_in_fat = measured_archs - set(fat_targets.targets)
                    if not_in_fat:
                        raise PatchCampaignError(
                            f"{args.patch}: measured architectures "
                            f"{sorted(not_in_fat)} are not in the requested "
                            f"fat targets {sorted(fat_targets.targets)}"
                        )


def _producer_gpu_count_preflight(args: argparse.Namespace, bound_contracts) -> None:
    # RD58 (PA36 migration #4, dev-gpt-agent req_82fbbafe52c0472d
    # Q4): generic pre-scaffold GPU-count preflight. A bound
    # contract that declares scope.gpu_count.minimum (RD58 is
    # currently the only one) is enforced BEFORE the expensive
    # 5-build scaffold: fail closed if HIP_VISIBLE_DEVICES does
    # not declare enough distinct selector tokens. Generic, not an
    # RD58-specific branch.
    required_gpu_count: int | None = None
    for _contract in bound_contracts:
        if _contract.scope.gpu_count is not None:
            _minimum = _contract.scope.gpu_count.minimum
            if _minimum is not None:
                required_gpu_count = (
                    _minimum
                    if required_gpu_count is None
                    else max(required_gpu_count, _minimum)
                )
    if required_gpu_count is not None:
        from bigcherry.experiment.execution import (
            require_device_visibility,
        )

        require_device_visibility(
            context=f"{args.patch} pre-scaffold GPU-count preflight",
            minimum_count=required_gpu_count,
        )


def _producer_campaign_identity_digest(
    args: argparse.Namespace,
    *,
    scaffold: StandardCampaignScaffold,
    patch_digest: str,
    subject_tree: str,
    base_revision: str,
) -> str:
    from bigcherry.patch import evidence as patch_validation_evidence

    model_identity = (
        _producer_file_identity(Path(args.model))
        if args.model is not None
        else None
    )
    corpus_identity = (
        _producer_file_identity(Path(args.producer_corpus))
        if args.producer_corpus is not None
        else None
    )
    if model_identity is None and corpus_identity is None:
        # RD12 pilot path: no model/corpus inputs -- the model-free
        # digest stays bit-identical to the approved pilot records.
        campaign_identity_digest = (
            patch_validation_evidence.model_free_campaign_identity_digest(
                patch_name=args.patch,
                patch_digest=patch_digest,
                patched_source_tree=subject_tree,
                gpu_architecture=args.amdgpu_targets,
                campaign_build_identities=scaffold.campaign_build_identities,
                base_revision=base_revision,
            )
        )
    else:
        # Input-bound identity (GPT review req_7a72896b609a48b5
        # BLOCKER #1): the producer's real model/corpus file facts are
        # hashed in, so two runs that differ in model or corpus bytes
        # never share a campaign identity.
        campaign_identity_digest = (
            patch_validation_evidence.producer_campaign_identity_digest(
                patch_name=args.patch,
                patch_digest=patch_digest,
                patched_source_tree=subject_tree,
                gpu_architecture=args.amdgpu_targets,
                campaign_build_identities=scaffold.campaign_build_identities,
                base_revision=base_revision,
                model=model_identity,
                corpus=corpus_identity,
            )
        )
    return campaign_identity_digest


def _producer_scaffold_build_apply_evidence(
    args: argparse.Namespace,
    *,
    scaffold: StandardCampaignScaffold,
    run_dir: Path,
    control_tree: str,
    subject_tree: str,
):
    """Write the scaffold build/apply artifacts; return (build_evidence, apply_evidence)."""
    build_evidence = {
        "control": {
            "build_id": scaffold.control_build_evidence.effective_build_id,
            "source_tree": control_tree,
            "architecture": args.amdgpu_targets,
            "options": scaffold.control_build_evidence.effective_configure,
            "compile_commands": _write_bound_artifact(
                run_dir,
                "build/control-compile-commands.json",
                scaffold.control_build_evidence.verification.to_dict(),
            ),
            "runtime_bundle": _write_bound_artifact(
                run_dir,
                "build/control-runtime-bundle.json",
                scaffold.control_build_evidence.runtime_artifacts,
            ),
        },
        "subject": {
            "build_id": scaffold.validation_subject_build_evidence.effective_build_id,
            "source_tree": subject_tree,
            "architecture": args.amdgpu_targets,
            "options": scaffold.validation_subject_build_evidence.effective_configure,
            "compile_commands": _write_bound_artifact(
                run_dir,
                "build/subject-compile-commands.json",
                scaffold.validation_subject_build_evidence.verification.to_dict(),
            ),
            "runtime_bundle": _write_bound_artifact(
                run_dir,
                "build/subject-runtime-bundle.json",
                scaffold.validation_subject_build_evidence.runtime_artifacts,
            ),
        },
    }

    apply_evidence = {
        "control": {
            "verified": True,
            "idempotent": scaffold.control_idempotent,
            "artifact": _write_bound_artifact(
                run_dir,
                "apply/control.json",
                {
                    "source_tree": control_tree,
                    "composition": list(scaffold.control_composition),
                },
            ),
        },
        "subject": {
            "verified": True,
            "idempotent": scaffold.subject_idempotent,
            "artifact": _write_bound_artifact(
                run_dir,
                "apply/subject.json",
                {
                    "source_tree": subject_tree,
                    "composition": list(scaffold.subject_composition),
                },
            ),
        },
    }
    return build_evidence, apply_evidence


def _prepare_standard_producer_campaign(
    args: argparse.Namespace,
    *,
    cfg,
    registry,
    descriptor,
    bound_contracts,
    fat_targets: FatTargetPlan,
    workdir: Path,
) -> _ProducerCampaignSetup:
    from bigcherry.patch import source as psi
    from bigcherry.patch import validation as patch_validation

    _producer_gpu_count_preflight(args, bound_contracts)
    baseline_source = getattr(args, "baseline_source", "bigcherry")
    scaffold = _build_standard_campaign_scaffold(
        patch_id=args.patch,
        base_ref=cfg.pinned,
        baseline_source=baseline_source,
        hip_path=args.hip_path,
        amdgpu_targets=args.amdgpu_targets,
        workdir=workdir,
        worktree_root=args.worktree_root,
        build_root=args.build_root,
    )

    base_revision = scaffold.base_revision
    run_dir = workdir / "campaign"
    run_dir.mkdir(parents=True, exist_ok=True)

    control_tree = psi.git_worktree_tree(scaffold.control_source)
    subject_tree = psi.git_worktree_tree(scaffold.subject_source)
    patch_file = registry.root / descriptor.implementation_path
    patch_digest = psi.patch_implementation_digest(args.patch)

    campaign_identity_digest = _producer_campaign_identity_digest(
        args,
        scaffold=scaffold,
        patch_digest=patch_digest,
        subject_tree=subject_tree,
        base_revision=base_revision,
    )
    build_evidence, apply_evidence = _producer_scaffold_build_apply_evidence(
        args,
        scaffold=scaffold,
        run_dir=run_dir,
        control_tree=control_tree,
        subject_tree=subject_tree,
    )

    package_root = (
        registry.root / descriptor.package_root
        if descriptor.package_root is not None
        else {}
    )
    validation_context = patch_validation.ValidationContext(
        descriptor=descriptor,
        base_revision=base_revision,
        control_source=scaffold.control_source,
        subject_source=scaffold.subject_source,
        stock_source=scaffold.stock_source,
        package_root=package_root,
        control_tree=control_tree,
        subject_tree=subject_tree,
        build_identities={
            "control": scaffold.control_build_evidence.effective_build_id,
            "subject": scaffold.validation_subject_build_evidence.effective_build_id,
        },
        build_evidence=build_evidence,
        apply_evidence=apply_evidence,
        architecture=args.amdgpu_targets,
        model=str(args.model) if args.model is not None else None,
        contracts=bound_contracts,
        contract_hashes={c.id: c.contract_hash for c in bound_contracts},
        run_dir=run_dir,
        register_artifact=patch_validation.make_default_register_artifact(run_dir),
        trace_evidence={},
        correctness_evidence={},
        performance_evidence={},
    )

    evidence_binding = ProducerEvidenceBindingContext(
        run_dir=run_dir,
        patch_id=args.patch,
        patch_path=patch_file,
        base_revision=base_revision,
        patched_source_tree=subject_tree,
        campaign_identity_digest=campaign_identity_digest,
        gpu_architectures=fat_targets.targets,
    )
    scaffold_validation_ids = scaffold.scaffold_validation_build_identities
    _scaffold_exe = ".exe" if sys.platform == "win32" else ""
    scaffold_validation_binaries = {
        "control": {
            "llama-bench": scaffold.control_bin / f"llama-bench{_scaffold_exe}",
            "llama-server": scaffold.control_bin / f"llama-server{_scaffold_exe}",
        },
        "subject": {
            "llama-bench": scaffold.validation_subject_bin
            / f"llama-bench{_scaffold_exe}",
            "llama-server": scaffold.validation_subject_bin
            / f"llama-server{_scaffold_exe}",
        },
    }
    return _ProducerCampaignSetup(
        scaffold=scaffold,
        evidence_binding=evidence_binding,
        campaign_identity_digest=campaign_identity_digest,
        base_revision=base_revision,
        run_dir=run_dir,
        validation_context=validation_context,
        scaffold_validation_ids=scaffold_validation_ids,
        scaffold_validation_binaries=scaffold_validation_binaries,
    )


def _prepare_self_contained_producer(
    *,
    cfg,
    descriptor,
    bound_contracts,
    workdir: Path,
    producer_id: str,
) -> _ProducerCampaignSetup:
    from bigcherry.patch import validation as patch_validation

    # Preserve current self-contained producer semantics.
    base_revision = cfg.pinned
    run_dir = workdir / "producer" / producer_id
    run_dir.mkdir(parents=True, exist_ok=True)
    validation_context = patch_validation.ValidationContext(
        descriptor=descriptor,
        base_revision=base_revision,
        control_source=None,
        subject_source=None,
        contracts=bound_contracts,
        contract_hashes={c.id: c.contract_hash for c in bound_contracts},
    )
    scaffold_validation_ids = {}
    scaffold_validation_binaries = {}
    return _ProducerCampaignSetup(
        scaffold=None,
        evidence_binding=None,
        campaign_identity_digest=None,
        base_revision=base_revision,
        run_dir=run_dir,
        validation_context=validation_context,
        scaffold_validation_ids=scaffold_validation_ids,
        scaffold_validation_binaries=scaffold_validation_binaries,
    )


def _execute_selected_producer(
    args: argparse.Namespace,
    *,
    producer_id: str,
    provided_inputs: Mapping[str, str],
    patch_dir: Path,
    workdir: Path,
    fat_targets: FatTargetPlan,
    device_map,
    selection,
    validation_plan,
    setup: _ProducerCampaignSetup,
):
    base_revision = setup.base_revision
    run_dir = setup.run_dir
    scaffold_validation_ids = setup.scaffold_validation_ids
    scaffold_validation_binaries = setup.scaffold_validation_binaries
    validation_context = setup.validation_context
    evidence_binding = setup.evidence_binding
    runtime = CampaignProducerRuntime(
        repo_root=REPO_ROOT,
        patch_id=args.patch,
        base_revision=base_revision,
        workdir=args.worktree_root,
        hip_path=args.hip_path,
        fat_targets=fat_targets,
        run_dir=run_dir,
    )
    producer_context = ProducerContext(
        repo_root=REPO_ROOT,
        patch_dir=patch_dir,
        workdir=workdir,
        campaign_id=f"{args.patch}/{producer_id}",
        base_revision=base_revision,
        hip_path=args.hip_path,
        fat_targets=fat_targets,
        model=args.model,
        corpus=args.producer_corpus,
        build_env=_hip_env(args.hip_path),
        inputs={},
        validation_build_identities=scaffold_validation_ids,
        patch_id=args.patch,
        device_map=device_map,
        runtime=runtime,
        validation_binaries=scaffold_validation_binaries,
    )
    return execute_validation_producer(
        patch_dir=patch_dir,
        producer_id=producer_id,
        provided_inputs=provided_inputs,
        producer_context=producer_context,
        validation_plan=validation_plan,
        validation_context=validation_context,
        correctness_evidence_requested=args.correctness_evidence is not None,
        performance_benchmark_requested=bool(args.run_performance_benchmark),
        selection=selection,
        evidence_binding_context=evidence_binding,
        bench_prompt=args.bench_prompt,
        bench_gen=args.bench_gen,
    )


def _producer_check_results(bound_contracts, execution):
    """Return (contract_correctness_gate, producer_check_results) for one execution."""
    # GPT review req_7a72896b609a48b5 BLOCKER #2: the real contract-
    # correctness gate over the producer's typed named results. The bound
    # contract is the authority on which named checks are required; the
    # producer only measures and reports. No producer results -> no gate
    # (the RD12 pilot record shape stays unchanged).
    contract_correctness_gate: dict[str, object] | None = None
    named = execution.result.contract_correctness_results
    if named:
        # Fail closed on the plural case (GPT re-review
        # req_6e79607f075c479b): the dispatcher is plural-aware, but the
        # named-result gate currently binds to a single contract's
        # authority. Rather than silently picking bound_contracts[0] when
        # several are bound, reject the ambiguous case. Duplicate check
        # names are rejected too -- the {check: result} mapping below
        # would otherwise silently overwrite one result with another.
        if len(bound_contracts) != 1:
            raise PatchCampaignError(
                "contract_correctness_results currently requires exactly "
                "one bound contract"
            )
        if len({result.check for result in named}) != len(named):
            raise PatchCampaignError(
                "contract_correctness_results contains duplicate check names"
            )
        contract_correctness_gate = compute_contract_correctness_gate(
            bound_contracts[0],
            {result.check: result for result in named},
        )
    producer_check_results = {
        check_id: asdict(result) for check_id, result in execution.evaluated.items()
    }
    if contract_correctness_gate is not None:
        producer_check_results = {
            **producer_check_results,
            "_contract_correctness_gate": contract_correctness_gate,
        }
    return contract_correctness_gate, producer_check_results


def _aggregate_producer_session_effects(
    args: argparse.Namespace,
    *,
    contract,
    contract_id: str,
    lane_effects,
    target_metric: str,
    aggregated,
    fat_targets: FatTargetPlan,
):
    # RD73 (PA36 migration #5, dev-gpt-agent req_a232ff7fb1f045db):
    # session aggregation (dispatcher-owned). Under a session
    # policy, the gain bound is established across repeated
    # SESSIONS, not from the pairs inside this one run. Fold the
    # prior sessions' persisted lane effects together with the one
    # just measured and re-aggregate over all of them.
    if (
        contract.acceptance.effect_evidence_policy
        == "session_ci95_threshold_bound_v1"
    ):
        from bigcherry.patch import evidence as patch_validation_evidence

        prior_records = patch_validation_evidence.load_records(args.patch)
        # GPT round 6 BLOCKER: schema-v4 persists contracts as a
        # list of {"id": ..., "hash": ...}, not a mapping.
        matching_records = [
            r
            for r in prior_records
            if any(
                entry.get("id") == contract_id
                and entry.get("hash") == contract.contract_hash
                for entry in r.get("contracts", [])
                if isinstance(entry, dict)
            )
        ]
        # GPT round 6 BLOCKER: use fat_targets.targets (not
        # ctx.amdgpu_targets which doesn't exist in this scope)
        session_archs = list(fat_targets.targets)
        # Build the current-session stub
        this_session = {
            "gpu_architectures": session_archs,
            "lane_effects": [
                {
                    "role": e.role,
                    "metric": e.metric,
                    "pair_ratios": list(e.pair_ratios),
                }
                for e in lane_effects
            ],
        }
        gain_field = (
            "end_to_end_gain_pct"
            if contract.acceptance.end_to_end_gain_pct is not None
            else "target_kernel_gain_pct"
        )
        aggregated = dict(aggregated)
        aggregated.update(
            experiment_contract.aggregate_session_effects(
                [*matching_records, this_session],
                field=gain_field,
                role="positive",
                metric=target_metric,
                architectures=session_archs,
            )
        )
    return aggregated


def _producer_resource_gate(contract, contract_id: str, execution):
    # RD73 (PA36 migration #5, dev-gpt-agent req_a232ff7fb1f045db):
    # compute the resource gate from the producer's
    # promotion_resource_results. A resource-bound contract
    # without resource evidence must fail closed.
    resource_gate = None
    if contract.acceptance.resource_limits:
        resource_results = execution.result.promotion_resource_results.get(
            contract_id
        )
        if not resource_results:
            raise PatchCampaignError(
                f"promotion_lane_effects for {contract_id!r} "
                "requires promotion_resource_results (a "
                "resource-bound contract without resource "
                "evidence must fail closed)"
            )
        # Build a {metric: result} mapping (evaluate_resource_gate
        # expects a dict, not a list). Reject duplicate metrics.
        resource_map: dict[str, experiment_contract.ResourceResult] = {}
        for rr in resource_results:
            if rr.metric in resource_map:
                raise PatchCampaignError(
                    f"promotion_resource_results for "
                    f"{contract_id!r}: duplicate metric "
                    f"{rr.metric!r}"
                )
            resource_map[rr.metric] = rr
        resource_gate = experiment_contract.evaluate_resource_gate(
            contract, resource_map
        )
    return resource_gate


def _evaluate_producer_contract_promotion(
    args: argparse.Namespace,
    *,
    contract,
    contract_id: str,
    lane_effects,
    execution,
    contract_correctness_gate,
    fat_targets: FatTargetPlan,
):
    target_metric = execution.result.promotion_target_metric.get(contract_id)
    if target_metric is None:
        raise PatchCampaignError(
            f"promotion_lane_effects for {contract_id!r} requires "
            "promotion_target_metric"
        )
    # RD58 (PA36 migration #4, dev-gpt-agent
    # req_ecb4b77a4c4e4bdd MAJOR #3): a producer that
    # supplies promotion lanes MUST have a matching
    # evaluated contract correctness gate -- missing
    # correctness evidence is BLOCKED/absent evidence, not a
    # measured promotion FAIL.
    if contract_correctness_gate is None:
        raise PatchCampaignError(
            f"promotion_lane_effects for {contract_id!r} "
            "requires an evaluated contract correctness gate "
            "(the producer supplied promotion lanes but no "
            "contract_correctness_results)"
        )
    # RD58 (PA36 migration #4, dev-gpt-agent
    # req_ecb4b77a4c4e4bdd BLOCKER): a producer that
    # supplies promotion lanes MUST supply trigger evidence --
    # a contract PASS without trigger proof would be a
    # fail-OPEN (the target code path may never have run).
    trigger_evidence = execution.result.promotion_trigger_evidence.get(
        contract_id
    )
    if not trigger_evidence:
        raise PatchCampaignError(
            f"promotion_lane_effects for {contract_id!r} "
            "requires promotion_trigger_evidence (a contract "
            "PASS without trigger proof is fail-OPEN)"
        )
    trigger_proof = experiment_contract.evaluate_trigger_proof(
        list(trigger_evidence)
    )
    # RD58 (PA36 migration #4, dev-gpt-agent
    # req_ecb4b77a4c4e4bdd MAJOR #3): require the
    # lane/metric keysets to match -- a mismatch would
    # silently aggregate the wrong lanes.
    lane_metrics = {e.metric for e in lane_effects}
    if target_metric not in lane_metrics:
        raise PatchCampaignError(
            f"promotion_lane_effects for {contract_id!r}: "
            f"target_metric {target_metric!r} not in lane "
            f"metrics {sorted(lane_metrics)}"
        )
    aggregated = experiment_contract.aggregate_contract_effects(
        contract, list(lane_effects), target_metric=target_metric
    )
    aggregated = _aggregate_producer_session_effects(
        args,
        contract=contract,
        contract_id=contract_id,
        lane_effects=lane_effects,
        target_metric=target_metric,
        aggregated=aggregated,
        fat_targets=fat_targets,
    )
    resource_gate = _producer_resource_gate(contract, contract_id, execution)
    return experiment_contract.evaluate_promotion_gate(
        contract,
        correctness_gate=contract_correctness_gate,
        aggregated_effects=aggregated,
        trigger_proof=trigger_proof,
        resource_gate=resource_gate,
    )


def _evaluate_producer_contract_promotions(
    args: argparse.Namespace,
    *,
    bound_contracts,
    execution,
    contract_correctness_gate,
    fat_targets: FatTargetPlan,
) -> dict[str, dict[str, object]]:
    # RD58 (PA36 migration #4, dev-gpt-agent req_82fbbafe52c0472d
    # Q6): the typed producer->dispatcher promotion channel. The
    # producer supplies per-contract lane_effects (real LaneEffect
    # objects) + target_metric; the dispatcher owns
    # aggregate_contract_effects() + evaluate_promotion_gate() (the
    # producer never computes a gate itself). No
    # promotion_lane_effects -> {} (every bound contract persists
    # its explicit BLOCKED "no promotion result produced" verdict,
    # exactly as before for non-promoting producers).
    bound_by_id = {c.id: c for c in bound_contracts}
    # RD58 (PA36 migration #4, dev-gpt-agent
    # req_918c7e1be6614f84 invariant): the contract-ID keysets
    # of promotion_lane_effects, promotion_target_metric, and
    # promotion_trigger_evidence must be identical (a
    # per-contract entry in one without a matching entry in
    # another is a producer bug, not a silent omission).
    _lane_keys = set(execution.result.promotion_lane_effects)
    _metric_keys = set(execution.result.promotion_target_metric)
    _trigger_keys = set(execution.result.promotion_trigger_evidence)
    if _lane_keys != _metric_keys or _lane_keys != _trigger_keys:
        raise PatchCampaignError(
            "promotion channel keysets must be identical: "
            f"lane_effects={sorted(_lane_keys)}, "
            f"target_metric={sorted(_metric_keys)}, "
            f"trigger_evidence={sorted(_trigger_keys)}"
        )
    producer_contract_promotions: dict[str, dict[str, object]] = {}
    for (
        contract_id,
        lane_effects,
    ) in execution.result.promotion_lane_effects.items():
        contract = bound_by_id.get(contract_id)
        if contract is None:
            raise PatchCampaignError(
                f"promotion_lane_effects for unknown contract {contract_id!r}"
            )
        producer_contract_promotions[contract_id] = _evaluate_producer_contract_promotion(
            args,
            contract=contract,
            contract_id=contract_id,
            lane_effects=lane_effects,
            execution=execution,
            contract_correctness_gate=contract_correctness_gate,
            fat_targets=fat_targets,
        )
    return producer_contract_promotions


def _persist_producer_validation_record(
    args: argparse.Namespace,
    *,
    cfg,
    registry,
    descriptor,
    validation_plan,
    scaffold: StandardCampaignScaffold,
    execution,
    producer_check_results,
    producer_contract_promotions,
    campaign_identity_digest: str,
    run_dir: Path,
):
    """Persist the tracked evidence record; return (record_path, validation_contract_verdicts)."""
    from bigcherry.patch import evidence as patch_validation_evidence
    from bigcherry.patch import source as psi

    # RD58 (PA36 migration #4, dev-gpt-agent
    # req_ecb4b77a4c4e4bdd additional invariant): persist the
    # exact promotion_lane_effects used for the verdict in the
    # record's lane_effects -- otherwise the contract promotion
    # is derived from ephemeral measurements that cannot be
    # audited from committed evidence.
    promotion_lane_json = tuple(
        asdict(effect)
        for effects in execution.result.promotion_lane_effects.values()
        for effect in effects
    )
    combined_lane_effects = execution.result.lane_effects + promotion_lane_json

    # Contract promotions (evaluate_promotion_gate() results) are a
    # different semantic type from the producer's check dispositions
    # (execution.contract_verdicts); the promotion APIs must never be
    # fed dispositions (req_f5ba56f4088e4742 finding #2).
    validation_contracts, validation_contract_verdicts = (
        build_contract_evidence_for_persistence(
            validation_plan.contracts,
            producer_contract_promotions,
        )
    )

    validation_record = patch_validation_evidence.make_record(
        patch_id=args.patch,
        patch_path=registry.root / descriptor.implementation_path,
        patch_implementation_digest=psi.patch_implementation_digest(args.patch),
        base_ref=cfg.pinned,
        base_revision=scaffold.base_revision,
        framework_baseline_digest=psi.composition_digest(
            scaffold.subject_composition
        ),
        patched_source_tree=psi.git_worktree_tree(scaffold.subject_source),
        gpu_architectures=args.amdgpu_targets,
        activation_evidence=(
            execution.activation_evidence
            if execution.activation_evidence is not None
            else execution.result.activation_evidence
        ),
        activation_disposition=execution.activation_disposition,
        correctness=execution.bound_correctness,
        campaign_identity_digest=campaign_identity_digest,
        build_identities=scaffold.campaign_build_identities,
        # Deliberately producer-owned isolated pair, NOT scaffold pair.
        validation_build_identities=execution.result.validation_build_identities,
        campaign_workdir=run_dir,
        producer_artifact_names=execution.selection.spec.artifact_names,
        check_results=producer_check_results,
        validation_eligible=compute_persisted_validation_eligible(
            descriptor,
            execution.verdict,
            producer_contract_promotions,
            activation_disposition=execution.activation_disposition,
            correctness=(
                dict(execution.bound_correctness)
                if execution.bound_correctness is not None
                else {}
            ),
        ),
        lane_effects=combined_lane_effects,
        representation=descriptor.representation,
        validation_implementation_digest=descriptor.validation_digest,
        contracts=validation_contracts,
        contract_verdicts=validation_contract_verdicts,
        baseline_composition={
            "source": getattr(args, "baseline_source", "bigcherry"),
            "base_revision": scaffold.base_revision,
            "patches": list(scaffold.control_composition),
        },
        control_composition={
            "base_revision": scaffold.base_revision,
            "patches": list(scaffold.control_composition),
        },
        subject_composition={
            "base_revision": scaffold.base_revision,
            "patches": list(scaffold.subject_composition),
        },
        control_tree=psi.git_worktree_tree(scaffold.control_source),
        subject_tree=psi.git_worktree_tree(scaffold.subject_source),
        stock_tree=psi.git_worktree_tree(scaffold.stock_source),
    )
    record_path = patch_validation_evidence.write_record(validation_record)
    return record_path, validation_contract_verdicts


def _write_producer_outcome(
    args: argparse.Namespace,
    *,
    producer_id: str,
    execution,
    producer_check_results,
    scaffold: StandardCampaignScaffold | None,
    validation_contract_verdicts,
    record_path: Path | None,
    run_dir: Path,
) -> None:
    """Write the workdir-local producer-execution.json outcome and print the verdict."""
    outcome_doc = {
        "patch_id": args.patch,
        "producer_id": producer_id,
        "eligible": execution.verdict.eligible,
        "reasons": list(execution.verdict.reasons),
        "blocked": execution.verdict.blocked,
        "errors": list(execution.verdict.errors),
        "check_results": producer_check_results,
        "contract_verdicts": (
            validation_contract_verdicts
            if scaffold is not None
            else dict(execution.contract_verdicts)
        ),
        "validation_build_identities": dict(
            execution.result.validation_build_identities
        ),
        "evidence_record": str(record_path) if record_path is not None else None,
    }
    outcome_path = run_dir / "producer-execution.json"
    _atomic_write_json(outcome_path, outcome_doc)

    _print(
        f"validation producer {args.patch}/{producer_id}: "
        f"{'eligible' if execution.verdict.eligible else 'ineligible'} "
        f"({len(execution.verdict.reasons)} blocking reasons) -- "
        f"{outcome_path}"
    )


def _resolve_producer_targets(args: argparse.Namespace, bound_contracts):
    """Return (fat_targets, device_map), validated against the bound contracts' scope."""
    amdgpu_targets = args.amdgpu_targets
    fat_targets = FatTargetPlan(
        targets=tuple(amdgpu_targets.split(";")) if amdgpu_targets else (),
    )
    device_map = _parse_producer_device_map(list(args.device_map or ()))

    _validate_producer_architectures(args, bound_contracts, fat_targets, device_map)
    return fat_targets, device_map


def _load_producer_plan(args: argparse.Namespace):
    """Return (registry, descriptor, cfg, validation_plan, bound_contracts)."""
    from bigcherry.core import paths as bc_paths
    from bigcherry.core import config as campaign_config
    from bigcherry.patch import registry as patch_registry
    from bigcherry.patch import validation as patch_validation
    from bigcherry.patch import validation_policy as patch_validation_policy

    registry = patch_registry.load_registry(bc_paths.PATCHES)
    descriptor = registry.get(args.patch)
    cfg = campaign_config.load(bc_paths.RECIPES)

    validation_plan = patch_validation_policy.require_execution_package(
        descriptor,
        root=bc_paths.PATCHES,
    )
    if validation_plan is None:
        raise PatchCampaignError(
            f"{args.patch}: --validation-producer requires a resolvable validation plan"
        )

    # Plural-aware (VA17): a --validation-producer patch may bind more than
    # one Experiment Contract (e.g. 1203's RD05/RD06/RD07) -- the singular
    # load_contract_for_descriptor()/.experiment_contract compatibility path
    # fails closed (PatchRegistryError) for exactly that case, so this
    # generic dispatcher must use the plural loader, never the singular one.
    bound_contracts = patch_validation.load_contracts_for_descriptor(descriptor)
    return registry, descriptor, cfg, validation_plan, bound_contracts


def _run_validation_producer(
    args: argparse.Namespace,
    *,
    producer_id: str,
    provided_inputs: Mapping[str, str],
) -> int:
    """PA36-F step 5 entry point: resolve+execute one patch-local producer
    through the generic dispatcher end to end, entirely outside the legacy
    tune/replay/stock campaign-build flow ``run()`` otherwise always runs --
    every build this path performs goes through
    ``CampaignProducerRuntime.build_pair()`` instead (the PA36 build-once-
    fat-multiarch authority), mirroring how ``_run_performance_benchmark()``
    is already its own self-contained path rather than a branch bolted onto
    the legacy one.

    Binding this producer's typed result into a persisted, tracked
    ``patch_validation_evidence`` record (the shape GPT's design section 3
    sketches for a caller) is real per-patch MIGRATION work -- it requires
    that migration's own campaign-build identity domain decisions (what
    ``build_identities`` even means for a patch with no tune/replay/stock
    build) and is explicitly out of PA36-F's scope (see the atomic
    migration sequence). This prints the verdict and writes the generic,
    typed execution outcome as a workdir-local JSON artifact -- the same
    diagnostic-artifact pattern ``--run-performance-benchmark`` already
    uses for ``performance-matrix.json`` -- and exits 0 iff the plan is
    eligible.
    """
    import os

    from bigcherry.core import paths as bc_paths

    os.environ["ROCM_PATH"] = str(args.hip_path)
    os.environ["HIP_PATH"] = str(args.hip_path)
    os.environ["PATH"] = os.pathsep.join(
        [str(args.hip_path / "bin"), os.environ.get("PATH", "")]
    )

    registry, descriptor, cfg, validation_plan, bound_contracts = (
        _load_producer_plan(args)
    )

    workdir: Path = args.workdir
    workdir.mkdir(parents=True, exist_ok=True)
    patch_dir = bc_paths.PATCHES / args.patch

    # Resolve before expensive scaffold work because manifest policy owns
    # whether a standard campaign is required.
    selection = resolve_producer(
        patch_dir=patch_dir,
        producer_id=producer_id,
    )
    if selection.spec.patch_id != args.patch:
        raise ValidationProducerError(
            f"producer patch_id {selection.spec.patch_id!r} does not match "
            f"requested patch {args.patch!r}"
        )

    # Fail fast before builds. execute_validation_producer() deliberately
    # repeats these authoritative gates for direct callers.
    validate_producer_inputs(selection.spec, provided_inputs)
    validate_producer_cli_compatibility(
        selection.spec,
        correctness_evidence_requested=args.correctness_evidence is not None,
        performance_benchmark_requested=bool(args.run_performance_benchmark),
    )

    fat_targets, device_map = _resolve_producer_targets(args, bound_contracts)

    if selection.spec.standard_campaign == "run":
        setup = _prepare_standard_producer_campaign(
            args,
            cfg=cfg,
            registry=registry,
            descriptor=descriptor,
            bound_contracts=bound_contracts,
            fat_targets=fat_targets,
            workdir=workdir,
        )
    else:
        # Preserve current self-contained producer semantics.
        setup = _prepare_self_contained_producer(
            cfg=cfg,
            descriptor=descriptor,
            bound_contracts=bound_contracts,
            workdir=workdir,
            producer_id=producer_id,
        )
    scaffold = setup.scaffold
    run_dir = setup.run_dir

    execution = _execute_selected_producer(
        args,
        producer_id=producer_id,
        provided_inputs=provided_inputs,
        patch_dir=patch_dir,
        workdir=workdir,
        fat_targets=fat_targets,
        device_map=device_map,
        selection=selection,
        validation_plan=validation_plan,
        setup=setup,
    )

    contract_correctness_gate, producer_check_results = _producer_check_results(
        bound_contracts, execution
    )

    record_path: Path | None = None
    validation_contract_verdicts = None
    if scaffold is not None:
        assert setup.campaign_identity_digest is not None
        producer_contract_promotions = _evaluate_producer_contract_promotions(
            args,
            bound_contracts=bound_contracts,
            execution=execution,
            contract_correctness_gate=contract_correctness_gate,
            fat_targets=fat_targets,
        )
        record_path, validation_contract_verdicts = _persist_producer_validation_record(
            args,
            cfg=cfg,
            registry=registry,
            descriptor=descriptor,
            validation_plan=validation_plan,
            scaffold=scaffold,
            execution=execution,
            producer_check_results=producer_check_results,
            producer_contract_promotions=producer_contract_promotions,
            campaign_identity_digest=setup.campaign_identity_digest,
            run_dir=run_dir,
        )

    _write_producer_outcome(
        args,
        producer_id=producer_id,
        execution=execution,
        producer_check_results=producer_check_results,
        scaffold=scaffold,
        validation_contract_verdicts=validation_contract_verdicts,
        record_path=record_path,
        run_dir=run_dir,
    )

    # Success means the requested producer execution and, when required,
    # tracked evidence persistence completed. Eligibility is evidence,
    # not process success (dev-gpt-agent req_2ecda033763949a9 T5).
    return 0
