"""HI81 (informal): unattended patch-validation campaign.

Given only a patch module name and a model path, materializes an isolated,
content-addressed source worktree for that one patch (patch_source_
isolation.py, HI82 -- never the shared vendor/llama.cpp tree), builds the
tune-mode/replay-mode/stock-baseline trees from it, then runs the full
record->tune->promote->export->replay->bench->report smoke campaign
(tools/bigcherry/e2e_smoke_campaign.py) against it -- the same process used
to validate HI30/HI31 this session, generalized to any single untested
patch so a human only has to choose the patch and the model.

This intentionally does NOT decide whether the patch is good -- it produces
the same report.md/bench.json/measurements.jsonl a human (or a follow-up
GPT review) reads to make that call. It also does not touch git or the
patch's catalog state (validated/rejected/untested) -- promoting a patch
out of "untested" is a separate, deliberate decision.

Usage:
    python -m bigcherry.patch.validation_campaign \\
        --patch <patch-id> \\
        --model $BC_MODEL_ROOT/qwen3.5-2b/Qwen_Qwen3.5-2B-Q4_K_M.gguf \\
        --hip-path vendor/rocm/7.1 --amdgpu-targets gfx1100 \\
        --manifest artifacts/<rev>/hip-autotune-manifest.json \\
        --workdir work/patch-1204-qwen2b

Safe to re-run: source materialization and every build/campaign stage below
reuse existing output where present (patch_source_isolation.py's manifest-
verified worktree reuse, cmake incremental builds, e2e_smoke_campaign.py's
per-stage resume-check).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import os
import re
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from bigcherry.build.builds import capture_completed_build_evidence
from bigcherry.core.context import ProjectContext
from bigcherry.core.paths import REPO_ROOT
from bigcherry.patch.activation import (
    verdict,
    write_activation_json,
)
from bigcherry.patch.campaign.benchmark import _run_performance_benchmark
from bigcherry.patch.campaign.build import (
    _atomic_write_json,
    _full_requested_cmake_args,
    _hip_env,
    _print,
    _write_bound_artifact,
    build_tree,
    generate_registry,
    PatchCampaignError,
)
from bigcherry.patch.campaign.contract import (
    build_contract_evidence_for_persistence,
    compute_contract_correctness_gate,
    compute_persisted_validation_eligible,
)
from bigcherry.patch.campaign.producer import (
    _parse_producer_inputs,
    _parse_validation_producer_selector,
    _run_validation_producer,
)
from bigcherry.patch.campaign.scaffold import _build_standard_campaign_scaffold
from bigcherry.patch.campaign.trace import run_trace_activation_probes


@dataclasses.dataclass(frozen=True)
class _FrameworkSource:
    baseline_source: str
    base_revision: str
    composition: tuple
    source: Path
    idempotent: bool
    source_tree: str
    source_manifest: dict
    source_identity: dict


def _require_framework_configuration_inputs(args: argparse.Namespace, descriptor) -> None:
    from bigcherry.patch import validation_policy

    if not validation_policy.is_framework_configuration_patch(descriptor):
        raise PatchCampaignError(
            "--framework-configuration requires a local packaged framework patch without an RD/contract binding"
        )
    if any(
        getattr(args, name, False)
        for name in (
            "correctness_evidence",
        )
    ):
        raise PatchCampaignError(
            "framework configuration cannot be combined with runtime qualification modes"
        )
    if args.amdgpu_targets is None:
        raise PatchCampaignError(
            "framework configuration requires explicit AMDGPU compile targets"
        )
    targets = tuple(
        target.strip()
        for target in re.split(r"[,;]", args.amdgpu_targets)
        if target.strip()
    )
    if not targets or any(
        not re.fullmatch(r"gfx[0-9a-f]+", target) for target in targets
    ):
        raise PatchCampaignError(
            "framework configuration requires explicit AMDGPU compile targets"
        )
    args.amdgpu_targets = ";".join(targets)


def _materialize_framework_source(args: argparse.Namespace, descriptor, cfg) -> _FrameworkSource:
    from bigcherry.patch import source as psi

    from bigcherry.core.context import ProjectContext

    base_repo = ProjectContext.resolve(
        work_root=os.environ.get("BC_CACHE")
    ).upstream_repo
    baseline_source = "bigcherry-qualification-tuning"
    base_revision, composition = psi.resolve_source_composition(
        baseline_source,
        focal=None,
        base_ref=cfg.pinned,
        base_repo=base_repo,
    )
    if (descriptor.patch_id, descriptor.implementation_digest) not in composition:
        raise PatchCampaignError(
            f"framework source {baseline_source!r} does not contain focal patch {descriptor.patch_id!r}"
        )
    source = psi.materialize_composition(
        base_repo=base_repo,
        worktree_root=args.worktree_root / "framework",
        resolved_revision=base_revision,
        composition=composition,
        overlay_root=psi.REPO_ROOT / "src",
        requested_revision=cfg.pinned,
    )
    idempotent = psi.verify_composition_idempotent(
        base_repo=base_repo,
        source=source,
        worktree_root=args.worktree_root / "framework",
        resolved_revision=base_revision,
        composition=composition,
        overlay_root=psi.REPO_ROOT / "src",
        requested_revision=cfg.pinned,
    )
    if not idempotent:
        raise PatchCampaignError("framework composition did not reapply idempotently")
    source_tree = psi.git_worktree_tree(source)
    source_manifest = psi._read_manifest(source)
    if not source_manifest or source_manifest.get("source_tree_oid") != source_tree:
        raise PatchCampaignError("framework source attestation is missing or stale")
    source_identity = psi._make_source_identity_v2(
        resolved_revision=base_revision,
        composition=composition,
        overlay_root=psi.REPO_ROOT / "src",
    )
    source_identity["materialization_plan_id"] = source_identity["source_key"]
    if any(source_manifest.get(key) != value for key, value in source_identity.items()):
        raise PatchCampaignError("framework materialization identity is stale")
    return _FrameworkSource(
        baseline_source=baseline_source,
        base_revision=base_revision,
        composition=composition,
        source=source,
        idempotent=idempotent,
        source_tree=source_tree,
        source_manifest=source_manifest,
        source_identity=source_identity,
    )


def _generate_framework_inputs(args: argparse.Namespace, source: Path):
    """Generate the registry into a fresh build-root; return (build_root, generated_dir, manifest)."""
    from bigcherry.build import generated_tree

    build_root = (args.build_root or args.workdir) / source.name
    # Qualification owns fresh directories, never retroactively attests a
    # historical build whose inputs were not observed during compilation.
    for role in ("production", "diagnostic"):
        if (build_root / f"framework-{role}").exists():
            raise PatchCampaignError(
                "framework qualification requires a fresh build-root; preserve the previous run"
            )
    generated_dir = build_root / "generated"
    generate_registry(
        source=source, amdgpu_targets=args.amdgpu_targets, generated_dir=generated_dir
    )
    # Same four compile inputs returned by catalog.emit().compile_input_paths;
    # JSON manifests contain timestamps and are not compiler inputs.
    compile_inputs = tuple(
        generated_dir / name
        for name in (
            "hip-autotune-registry.inc",
            "hip-autotune-build-hash.h",
            "hip-autotune-arch.h",
            "hip-autotune-mmvq-instances.inc",
        )
    )
    missing = [str(path) for path in compile_inputs if not path.is_file()]
    if missing:
        raise PatchCampaignError(f"generated compiler inputs missing: {missing}")
    generated_manifest = generated_tree.build_manifest(
        generated_dir, compile_inputs=compile_inputs
    )
    return build_root, generated_dir, generated_manifest


def _compile_framework_builds(
    args: argparse.Namespace,
    *,
    source: Path,
    source_tree: str,
    build_root: Path,
    generated_dir: Path,
    generated_manifest: dict,
):
    """Build production/diagnostic trees; return (proof, production, diagnostic, compiler_observations)."""
    from bigcherry.build import generated_tree
    from bigcherry.patch import source as psi

    proof = {}

    def generated_proof(phase, build_dir):
        compiled_copy = build_dir / "generated-inputs"
        if psi.git_worktree_tree(source) != source_tree:
            raise PatchCampaignError(f"source changed at {phase}")
        generated_tree.verify_tree(generated_dir, generated_manifest)
        generated_tree.verify_tree(compiled_copy, generated_manifest)
        copied_manifest = generated_tree.build_manifest(
            compiled_copy,
            compile_inputs=tuple(
                compiled_copy / name for name in generated_manifest["compile_inputs"]
            ),
        )
        if (
            copied_manifest["compile_inputs_hash"]
            != generated_manifest["compile_inputs_hash"]
        ):
            raise PatchCampaignError(
                f"{build_dir.name}: compiled-copy input hash disagrees with generated manifest"
            )
        proof[build_dir.name] = copied_manifest

    import shutil

    for role in ("production", "diagnostic"):
        shutil.copytree(
            generated_dir, build_root / f"framework-{role}" / "generated-inputs"
        )
    common = [
        "-DGGML_HIP_RCCL=ON",
        "-DGGML_HIP_DISPATCH_REPLAY=ON",
        "-DGGML_HIP_AUTOTUNE=OFF",
        "-DGGML_HIP_AUTOTUNE_RECORD=OFF",
        "-DGGML_HIP_REPLAY_DIAGNOSTICS=OFF",
    ]
    production_args = common + [
        "-DGGML_HIP_DISPATCH_DIAGNOSTICS=OFF",
        f"-DGGML_HIP_AUTOTUNE_GENERATED_DIR={build_root / 'framework-production' / 'generated-inputs'}",
    ]
    diagnostic_args = common + [
        "-DGGML_HIP_DISPATCH_DIAGNOSTICS=ON",
        f"-DGGML_HIP_AUTOTUNE_GENERATED_DIR={build_root / 'framework-diagnostic' / 'generated-inputs'}",
    ]
    production_bin = build_tree(
        name="framework-production",
        hip_path=args.hip_path,
        amdgpu_targets=args.amdgpu_targets,
        workdir=build_root,
        targets=["llama-server"],
        source=source,
        extra_cmake_args=production_args,
        generated_proof_callback=generated_proof,
    )
    diagnostic_bin = build_tree(
        name="framework-diagnostic",
        hip_path=args.hip_path,
        amdgpu_targets=args.amdgpu_targets,
        workdir=build_root,
        targets=["llama-server"],
        source=source,
        extra_cmake_args=diagnostic_args,
        generated_proof_callback=generated_proof,
    )
    env = _hip_env(args.hip_path)
    exe = ".exe" if sys.platform == "win32" else ""
    production = capture_completed_build_evidence(
        build_root / "framework-production",
        source_root=source,
        architecture=args.amdgpu_targets,
        binary=production_bin / f"llama-server{exe}",
        requested_cmake_args=_full_requested_cmake_args(
            hip_path=args.hip_path,
            amdgpu_targets=args.amdgpu_targets,
            extra_cmake_args=production_args,
        ),
        build_env=env,
    )
    diagnostic = capture_completed_build_evidence(
        build_root / "framework-diagnostic",
        source_root=source,
        architecture=args.amdgpu_targets,
        binary=diagnostic_bin / f"llama-server{exe}",
        requested_cmake_args=_full_requested_cmake_args(
            hip_path=args.hip_path,
            amdgpu_targets=args.amdgpu_targets,
            extra_cmake_args=diagnostic_args,
        ),
        build_env=env,
    )
    from bigcherry.build.builds import inspect_dispatch_build

    compiler_observations = {}
    for role, diagnostic_on in (("production", False), ("diagnostic", True)):
        observed = inspect_dispatch_build(build_root / f"framework-{role}")
        counts = observed["compiled_definition_counts"]
        if (
            observed["issues"]
            or bool(counts["GGML_HIP_DISPATCH_DIAGNOSTICS"]) != diagnostic_on
        ):
            raise PatchCampaignError(
                f"{role} diagnostic compiler state disagrees with qualification role"
            )
        compiler_observations[role] = {
            key: observed[key]
            for key in (
                "hip_compile_command_count",
                "compiled_definition_counts",
                "coverage_translation_unit",
                "issues",
            )
        }
    return proof, production, diagnostic, compiler_observations


def _record_framework_configuration(
    args: argparse.Namespace,
    descriptor,
    cfg,
    framework: _FrameworkSource,
    *,
    build_root: Path,
    generated_manifest: dict,
    proof: dict,
    production,
    diagnostic,
    compiler_observations: dict,
) -> int:
    """Evaluate the framework checks and persist the schema-5 evidence record."""
    from bigcherry.patch import evidence as patch_validation_evidence
    from bigcherry.patch import validation_policy
    from bigcherry.patch import validation
    from bigcherry.core import paths as bc_paths

    base_revision = framework.base_revision
    composition = framework.composition
    idempotent = framework.idempotent
    source_tree = framework.source_tree
    source_manifest = framework.source_manifest
    source_identity = framework.source_identity
    baseline_source = framework.baseline_source
    run_dir = args.workdir / "framework" / descriptor.patch_id
    run_dir.mkdir(parents=True, exist_ok=False)
    generated_artifact = _write_bound_artifact(
        run_dir, "generated-tree.json", generated_manifest
    )
    source_artifact = _write_bound_artifact(
        run_dir, "source-tree.json", source_manifest
    )
    builds = {
        "production": production.campaign_identity(),
        "diagnostic": diagnostic.campaign_identity(),
    }
    for role in builds:
        compiler_observations[role]["build_identity"] = builds[role]
    build_artifacts = {
        role: _write_bound_artifact(
            run_dir,
            f"{role}-build.json",
            {
                **completed.to_dict(),
                "generated_inputs_verification": "compiled-copy-v1",
                "generated_inputs": proof[f"framework-{role}"],
                "source_slice_id": source_manifest["source_slice_id"],
                "source_tree": source_tree,
            },
        )
        for role, completed in (("production", production), ("diagnostic", diagnostic))
    }
    plan = validation_policy.require_execution_package(
        descriptor, root=bc_paths.PATCHES
    )
    ctx = validation.ValidationContext(
        descriptor=descriptor,
        base_revision=base_revision,
        control_source=None,
        subject_source=None,
        package_root=bc_paths.PATCHES / descriptor.package_root,
        run_dir=run_dir,
        register_artifact=validation.make_default_register_artifact(run_dir),
        configuration_evidence={
            "apply": {
                "single_composition": True,
                "verified": True,
                "idempotent": idempotent,
                "artifact": source_artifact,
            },
            "builds": {
                role: {"completed": True, "artifact": artifact}
                for role, artifact in build_artifacts.items()
            },
        },
    )
    results = {
        spec.check_id: validation.evaluate_check(spec, ctx) for spec in plan.checks
    }
    verdict = validation.compute_verdict(plan, results)
    _print(f"adapter eligible: {verdict.eligible}")
    for check_id, result in results.items():
        _print(f"{check_id}: {result.status}: {result.summary}")
    checks = {name: asdict(result) for name, result in results.items()}
    artifacts = {
        artifact.path: artifact.sha256
        for result in results.values()
        for artifact in result.artifacts
    }
    artifacts[generated_artifact["path"]] = generated_artifact["sha256"]
    record = patch_validation_evidence.make_framework_configuration_record(
        descriptor=descriptor,
        patch_path=bc_paths.PATCHES / descriptor.implementation_path,
        base_ref=cfg.pinned,
        base_revision=base_revision,
        source_name=baseline_source,
        source_composition=composition,
        source_tree=source_tree,
        source_slice_id=source_manifest["source_slice_id"],
        compiled_targets=tuple(
            target.strip()
            for target in re.split(r"[,;]", args.amdgpu_targets)
            if target.strip()
        ),
        builds=builds,
        source_identity=source_identity,
        compiler_observations=compiler_observations,
        generated_inputs={
            role: {
                "proof": "compiled-copy-v1",
                "compile_inputs_hash": proof[f"framework-{role}"][
                    "compile_inputs_hash"
                ],
                "tree_manifest": proof[f"framework-{role}"],
                "build_identity": builds[role],
            }
            for role in builds
        },
        check_results=checks,
        artifact_hashes=artifacts,
        campaign_workdir=run_dir,
    )
    path = patch_validation_evidence.write_record(record)
    _print(f"framework configuration evidence: {path}")
    return 0 if record["eligible_for_validated_state"] else 1


def _run_framework_configuration(args: argparse.Namespace, descriptor, cfg) -> int:
    """Build the canonical native framework composition and persist schema-5 proof."""
    _require_framework_configuration_inputs(args, descriptor)
    framework = _materialize_framework_source(args, descriptor, cfg)
    build_root, generated_dir, generated_manifest = _generate_framework_inputs(
        args, framework.source
    )
    proof, production, diagnostic, compiler_observations = _compile_framework_builds(
        args,
        source=framework.source,
        source_tree=framework.source_tree,
        build_root=build_root,
        generated_dir=generated_dir,
        generated_manifest=generated_manifest,
    )
    return _record_framework_configuration(
        args,
        descriptor,
        cfg,
        framework,
        build_root=build_root,
        generated_manifest=generated_manifest,
        proof=proof,
        production=production,
        diagnostic=diagnostic,
        compiler_observations=compiler_observations,
    )


def _prepare_standard_campaign(args: argparse.Namespace, st: SimpleNamespace) -> None:
    """PA43 run() stage. standard-campaign stage: materialize the five standard builds (via the shared
    scaffold), bind the campaign identity and construct the S1-S7 Campaign."""
    cfg = st.cfg
    psi = st.psi
    workdir = st.workdir
    worktree_root: Path = args.worktree_root
    # RV80/B6: the baseline is the source's EXPLICIT named composition from
    # config/recipes.toml (never the retired implicit state=='validated'
    # scan), resolved through the exact-composition validator; the base ref
    # resolves to an immutable SHA that enters the v2 source identity.
    baseline_source = getattr(args, "baseline_source", "bigcherry")
    # PA36 sub-slice 2 (dev-gpt-agent req_2ecda033763949a9, T2): the five
    # standard-campaign builds (tune -> replay -> stock -> control ->
    # validation-subject) live in _build_standard_campaign_scaffold(),
    # shared with the generic standard_campaign="run" producer path. Local
    # aliases below keep the rest of run() structurally unchanged.
    scaffold = _build_standard_campaign_scaffold(
        patch_id=args.patch,
        base_ref=cfg.pinned,
        baseline_source=baseline_source,
        hip_path=args.hip_path,
        amdgpu_targets=args.amdgpu_targets,
        workdir=workdir,
        worktree_root=worktree_root,
        build_root=args.build_root,
    )
    base_revision = scaffold.base_revision
    control_composition = scaffold.control_composition
    subject_composition = scaffold.subject_composition
    control_src = scaffold.control_source
    patched_src = scaffold.subject_source
    stock_src = scaffold.stock_source
    control_idempotent = scaffold.control_idempotent
    subject_idempotent = scaffold.subject_idempotent
    build_root = scaffold.build_root
    build_env = scaffold.build_env
    tune_bin = scaffold.tune_bin
    replay_bin = scaffold.replay_bin
    stock_bin = scaffold.stock_bin
    control_bin = scaffold.control_bin
    validation_subject_bin = scaffold.validation_subject_bin
    tune_build_evidence = scaffold.tune_build_evidence
    replay_build_evidence = scaffold.replay_build_evidence
    stock_build_evidence = scaffold.stock_build_evidence
    control_build_evidence = scaffold.control_build_evidence
    validation_subject_build_evidence = scaffold.validation_subject_build_evidence
    exe = ".exe" if sys.platform == "win32" else ""

    from bigcherry.e2e_smoke_campaign import (  # noqa: E402
        Campaign,
        CampaignError,
        CampaignIdentityContext,
    )

    # Hoisted: HI83's evidence record (below) needs the same values.
    patch_digest = psi.patch_implementation_digest(args.patch)
    control_source_tree = psi.git_worktree_tree(control_src)
    patched_source_tree = psi.git_worktree_tree(patched_src)

    identity_context = CampaignIdentityContext(
        patch_name=args.patch,
        patch_digest=patch_digest,
        patched_source_tree=patched_source_tree,
        gpu_architecture=args.amdgpu_targets,
        build_identities={
            "tune": tune_build_evidence.campaign_identity(),
            "replay": replay_build_evidence.campaign_identity(),
            "stock": stock_build_evidence.campaign_identity(),
        },
    )

    campaign = Campaign(
        model=args.model,
        tune_server=tune_bin / f"llama-server{exe}",
        replay_server=replay_bin / f"llama-server{exe}",
        manifest=args.manifest,
        workdir=workdir / "campaign",
        stock_bench=stock_bin / f"llama-bench{exe}",
        tune_bench=tune_bin / f"llama-bench{exe}",
        replay_bench=replay_bin / f"llama-bench{exe}",
        bench_prompt=args.bench_prompt,
        bench_gen=args.bench_gen,
        bench_repetitions=args.bench_repetitions,
        identity_context=identity_context,
    )

    # Bind the workdir before writing any patch-specific activation
    # evidence. campaign.run() will check it again; this earlier call
    # prevents a trace probe from writing evidence into a stale/mismatched
    # campaign directory.
    campaign.ensure_campaign_identity()
    st.CampaignError = CampaignError
    st.base_revision = base_revision
    st.baseline_source = baseline_source
    st.campaign = campaign
    st.control_bin = control_bin
    st.control_build_evidence = control_build_evidence
    st.control_composition = control_composition
    st.control_idempotent = control_idempotent
    st.control_source_tree = control_source_tree
    st.control_src = control_src
    st.exe = exe
    st.identity_context = identity_context
    st.patch_digest = patch_digest
    st.patched_source_tree = patched_source_tree
    st.patched_src = patched_src
    st.stock_src = stock_src
    st.subject_composition = subject_composition
    st.subject_idempotent = subject_idempotent
    st.tune_bin = tune_bin
    st.validation_subject_bin = validation_subject_bin
    st.validation_subject_build_evidence = validation_subject_build_evidence


def _run_activation_probe_stage(args: argparse.Namespace, st: SimpleNamespace) -> None:
    """PA43 run() stage. Activation stage: resolve the trace-marker check and run the generic
    positive/negative trace probe, binding its logs as trace evidence."""
    campaign = st.campaign
    exe = st.exe
    tune_bin = st.tune_bin
    validation_plan = st.validation_plan
    workdir = st.workdir
    activation_evidence = None
    activation_verdict = None
    trace_marker_regex = args.trace_marker_regex
    trace_description = args.trace_description
    if validation_plan is not None:
        trace_specs = tuple(
            spec
            for spec in validation_plan.checks
            if spec.capability == "activation" and spec.validator == "trace-marker"
        )
        if len(trace_specs) > 1:
            raise PatchCampaignError(
                f"{args.patch}: validation plan declares multiple trace-marker activation checks"
            )
        if trace_specs:
            configured_marker = trace_specs[0].config.get("marker-regex")
            if not isinstance(configured_marker, str) or not configured_marker:
                raise PatchCampaignError(
                    f"{args.patch}: trace-marker activation check has no marker-regex"
                )
            if (
                trace_marker_regex is not None
                and trace_marker_regex != configured_marker
            ):
                raise PatchCampaignError(
                    f"{args.patch}: CLI trace marker conflicts with validation.toml"
                )
            trace_marker_regex = configured_marker
            trace_description = trace_description or f"{args.patch} activation"
        elif trace_marker_regex is not None or trace_description is not None:
            raise PatchCampaignError(
                f"{args.patch}: trace CLI options require a trace-marker validation check"
            )
    trace_result = run_trace_activation_probes(
        marker_regex=trace_marker_regex,
        description=trace_description,
        binary=tune_bin / f"llama-bench{exe}",
        model=args.model,
        hip_path=args.hip_path,
        workdir=workdir / "campaign",
        bench_prompt=args.bench_prompt,
        bench_gen=args.bench_gen,
    )
    # VA11A: real bound trace_evidence for ValidationContext (was {}
    # unconditionally, which made _builtin_trace_marker always BLOCKED --
    # GPT round-7 review, req_3d12aa6668b14bb1). Built from the real
    # positive/negative probe logs run_trace_activation_probes() already
    # wrote to workdir/"campaign"/"logs"/... -- the same directory
    # ValidationContext.run_dir (campaign_run_dir) resolves bound artifacts
    # against, so no copy is needed, only a path+sha256 reference.
    trace_evidence: dict[str, object] = {}
    if trace_result is not None:
        activation_evidence, trace_detail = trace_result
        # This stage establishes activation evidence only -- the existing
        # patch_activation verdict contract accepts correctness_passed=None;
        # a later patch-class-specific correctness check can strengthen
        # this without changing the trace-probe mechanism.
        activation_verdict = verdict(activation_evidence, correctness_passed=None)
        write_activation_json(
            workdir / "campaign" / "activation.json",
            activation_evidence,
            activation_verdict,
            extra={
                "campaign_identity_digest": campaign.campaign_identity_digest,
                "trace_probe": trace_detail,
            },
        )
        _print(
            f"activation: {activation_evidence.status} ({activation_evidence.mechanism})"
        )

        def _bind_existing(relative_log_path: str) -> dict[str, str]:
            target = (workdir / "campaign" / relative_log_path).resolve()
            return {
                "path": relative_log_path,
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            }

        trace_evidence = {
            "positive": {
                "marker_regex": trace_detail["marker_regex"],
                "artifact": _bind_existing(trace_detail["positive"]["log"]),
            },
            "negative": {
                "marker_regex": trace_detail["marker_regex"],
                "artifact": _bind_existing(trace_detail["negative_control"]["log"]),
            },
        }
    st.activation_evidence = activation_evidence
    st.activation_verdict = activation_verdict
    st.trace_evidence = trace_evidence
    st.trace_marker_regex = trace_marker_regex


def _collect_build_and_correctness_evidence(args: argparse.Namespace, st: SimpleNamespace) -> None:
    """PA43 run() stage. Bind correctness, build and apply evidence for the validation record."""
    base_revision = st.base_revision
    campaign = st.campaign
    control_build_evidence = st.control_build_evidence
    control_composition = st.control_composition
    control_idempotent = st.control_idempotent
    control_source_tree = st.control_source_tree
    descriptor = st.descriptor
    patched_source_tree = st.patched_source_tree
    registry = st.registry
    subject_composition = st.subject_composition
    subject_idempotent = st.subject_idempotent
    validation_subject_build_evidence = st.validation_subject_build_evidence
    workdir = st.workdir
    # HI83: record what this campaign proved (or didn't), tracked so
    # STATE="validated" can eventually be checked against it. This is
    # purely additive evidence production -- it does not gate anything in
    # this campaign, and nothing in bigcherry apply/build consumes it yet
    # (see plan item HI83's notes for why hard enforcement is deliberately
    # deferred). A campaign with no correctness evidence and/or no
    # activation probe for this patch still writes a real record; it is
    # simply not eligible_for_validated_state.
    from bigcherry.patch import evidence as patch_validation_evidence  # noqa: E402

    # cfg is loaded once, above, before source resolution -- reused here
    # (was previously loaded a second time in this exact spot, after
    # source materialization had already resolved against "HEAD").
    # RS04: the evidence record's patch file path resolves through the
    # registry descriptor (flat or packaged) -- no f"{patch_id}.py" guessing
    # in this caller.
    _descriptor = descriptor
    _patch_file = registry.root / _descriptor.implementation_path
    campaign_run_dir = workdir / "campaign"
    correctness_summary = None
    # VA11A: ctx.correctness_evidence must carry a BOUND artifact reference
    # ({"artifact": {"path", "sha256"}}) -- _builtin_backend_ops reads via
    # ctx.correctness_evidence.get("artifact"), and a raw decoded dict with
    # no "artifact" key made every backend-ops check unconditionally BLOCKED
    # (real bug, confirmed by reading validation.py::_builtin_backend_ops
    # before this fix -- GPT round-7 review, req_3d12aa6668b14bb1).
    correctness_evidence: dict[str, object] = {}
    performance_evidence: dict[str, object] = {}
    if args.correctness_evidence is not None:
        correctness_summary = patch_validation_evidence.load_correctness_summary(
            args.correctness_evidence,
            patch_id=args.patch,
            subject_digest=patch_validation_evidence.patch_validation_subject_digest(
                _patch_file
            ),
            base_revision=base_revision,
            patched_source_tree=patched_source_tree,
            campaign_identity_digest=campaign.campaign_identity_digest,
            gpu_architectures=(args.amdgpu_targets,),
        )
        correctness_path = campaign_run_dir / "correctness.json"
        _atomic_write_json(correctness_path, correctness_summary)
        correctness_evidence = {
            "artifact": {
                "path": correctness_path.relative_to(campaign_run_dir).as_posix(),
                "sha256": hashlib.sha256(correctness_path.read_bytes()).hexdigest(),
            }
        }
    build_evidence = {
        "control": {
            "build_id": control_build_evidence.effective_build_id,
            "source_tree": control_source_tree,
            "architecture": args.amdgpu_targets,
            "options": control_build_evidence.effective_configure,
            "compile_commands": _write_bound_artifact(
                campaign_run_dir,
                "build/control-compile-commands.json",
                control_build_evidence.verification.to_dict(),
            ),
            "runtime_bundle": _write_bound_artifact(
                campaign_run_dir,
                "build/control-runtime-bundle.json",
                control_build_evidence.runtime_artifacts,
            ),
        },
        "subject": {
            "build_id": validation_subject_build_evidence.effective_build_id,
            "source_tree": patched_source_tree,
            "architecture": args.amdgpu_targets,
            "options": validation_subject_build_evidence.effective_configure,
            "compile_commands": _write_bound_artifact(
                campaign_run_dir,
                "build/subject-compile-commands.json",
                validation_subject_build_evidence.verification.to_dict(),
            ),
            "runtime_bundle": _write_bound_artifact(
                campaign_run_dir,
                "build/subject-runtime-bundle.json",
                validation_subject_build_evidence.runtime_artifacts,
            ),
        },
    }
    apply_evidence = {
        "control": {
            "verified": True,
            "idempotent": control_idempotent,
            "artifact": _write_bound_artifact(
                campaign_run_dir,
                "apply/control.json",
                {
                    "source_tree": control_source_tree,
                    "composition": list(control_composition),
                },
            ),
        },
        "subject": {
            "verified": True,
            "idempotent": subject_idempotent,
            "artifact": _write_bound_artifact(
                campaign_run_dir,
                "apply/subject.json",
                {
                    "source_tree": patched_source_tree,
                    "composition": list(subject_composition),
                },
            ),
        },
    }
    st._descriptor = _descriptor
    st._patch_file = _patch_file
    st.apply_evidence = apply_evidence
    st.build_evidence = build_evidence
    st.campaign_run_dir = campaign_run_dir
    st.correctness_evidence = correctness_evidence
    st.correctness_summary = correctness_summary
    st.patch_validation_evidence = patch_validation_evidence
    st.performance_evidence = performance_evidence


def _evaluate_validation_plan(args: argparse.Namespace, st: SimpleNamespace) -> None:
    """PA43 run() stage. Evaluate the declared validation plan and the contract correctness gate."""
    apply_evidence = st.apply_evidence
    base_revision = st.base_revision
    build_evidence = st.build_evidence
    campaign_run_dir = st.campaign_run_dir
    control_build_evidence = st.control_build_evidence
    control_source_tree = st.control_source_tree
    control_src = st.control_src
    correctness_evidence = st.correctness_evidence
    descriptor = st.descriptor
    patch_validation = st.patch_validation
    patched_source_tree = st.patched_source_tree
    patched_src = st.patched_src
    performance_evidence = st.performance_evidence
    registry = st.registry
    stock_src = st.stock_src
    trace_evidence = st.trace_evidence
    validation_plan = st.validation_plan
    validation_subject_build_evidence = st.validation_subject_build_evidence
    validation_check_results: dict[str, object] = {}
    validation_verdict = None
    if validation_plan is not None:
        # VA11A: package_root lets a packaged patch's custom validator
        # actually resolve its check(ctx) file (was always None -- any
        # custom check would fail closed for every packaged RD patch).
        package_root = (
            (registry.root / descriptor.package_root)
            if descriptor.package_root is not None
            else {}
        )
        # VA15 real-hardware finding: validation_plan.contract is a
        # patch_validation.ContractBinding -- a lightweight PROJECTION
        # (contract_id/hash/expected_effect/etc) that deliberately does NOT
        # carry .correctness/.acceptance/etc. compute_contract_correctness_gate()
        # needs the real experiment_contract.ExperimentContract, loaded
        # fresh here. PA36-F step 2: also the
        # source of ValidationContext's plural contracts/contract_hashes
        # below -- loaded once, before the context is constructed.
        full_contract = patch_validation.load_contract_for_descriptor(descriptor)
        validation_ctx = patch_validation.ValidationContext(
            descriptor=descriptor,
            base_revision=base_revision,
            control_source=control_src,
            subject_source=patched_src,
            stock_source=stock_src,
            package_root=package_root,
            control_tree=control_source_tree,
            subject_tree=patched_source_tree,
            # GPT round 3: RD58's real build check must be evaluated
            # against ITS test-save-load-state builds, not the generic
            # llama-bench builds the final record no longer identifies it
            # with.
            build_identities={
                "control": control_build_evidence.effective_build_id,
                "subject": validation_subject_build_evidence.effective_build_id,
            },
            build_evidence=build_evidence,
            apply_evidence=apply_evidence,
            architecture=args.amdgpu_targets,
            model=str(args.model),
            contracts=(full_contract,) if full_contract is not None else (),
            contract_hashes=(
                {full_contract.id: full_contract.contract_hash}
                if full_contract is not None
                else {}
            ),
            run_dir=campaign_run_dir,
            register_artifact=patch_validation.make_default_register_artifact(
                campaign_run_dir
            ),
            trace_evidence=trace_evidence,
            correctness_evidence=correctness_evidence,
            performance_evidence=performance_evidence,
        )
        evaluated = {
            spec.check_id: patch_validation.evaluate_check(spec, validation_ctx)
            for spec in validation_plan.checks
        }
        validation_verdict = patch_validation.compute_verdict(
            validation_plan, evaluated
        )
        # GPT round 2 (req_3616cc1d90dc4512, blocker #3): RD58's own real
        # test-save-load-state evidence produces a named
        # state_restore_integrity CorrectnessResult -- thread it through
        # here so the contract's own correctness gate actually reflects
        # the real evidence instead of reporting missing_checks.
        # The legacy run() path produces no named correctness results
        # (RD08/RD73 evidence flows only through the generic
        # --validation-producer path). An empty mapping preserves the gate's
        # missing_checks fail-closed behavior for any bound contract that
        # requires a named check.
        contract_correctness_gate = compute_contract_correctness_gate(
            full_contract, {}
        )
        validation_check_results = {
            check_id: asdict(result) for check_id, result in evaluated.items()
        }
        if contract_correctness_gate is not None:
            validation_check_results["_contract_correctness_gate"] = (
                contract_correctness_gate
            )
            _print(
                f"contract correctness gate: "
                f"{'passed' if contract_correctness_gate.get('passed') else contract_correctness_gate.get('status', 'not passed')}"
            )
        _print(
            f"validation verdict: {'eligible' if validation_verdict.eligible else 'ineligible'} "
            f"({len(validation_verdict.reasons)} blocking reasons)"
        )
    st.validation_check_results = validation_check_results
    st.validation_verdict = validation_verdict


def _persist_validation_record(args: argparse.Namespace, st: SimpleNamespace) -> int:
    """PA43 run() stage. Record-persistence stage: write the HI83 validation evidence record."""
    _descriptor = st._descriptor
    _patch_file = st._patch_file
    activation_evidence = st.activation_evidence
    activation_verdict = st.activation_verdict
    base_revision = st.base_revision
    baseline_source = st.baseline_source
    campaign = st.campaign
    cfg = st.cfg
    control_build_evidence = st.control_build_evidence
    control_composition = st.control_composition
    control_source_tree = st.control_source_tree
    correctness_summary = st.correctness_summary
    identity_context = st.identity_context
    patch_digest = st.patch_digest
    patch_validation_evidence = st.patch_validation_evidence
    patched_source_tree = st.patched_source_tree
    psi = st.psi
    stock_src = st.stock_src
    subject_composition = st.subject_composition
    validation_check_results = st.validation_check_results
    validation_plan = st.validation_plan
    validation_subject_build_evidence = st.validation_subject_build_evidence
    validation_verdict = st.validation_verdict
    workdir = st.workdir
    # The legacy run() path runs no contract qualification (that happens only
    # through --validation-producer), so no bound contract has a promotion
    # result here; compute_persisted_validation_eligible() therefore reports
    # a bound-contract patch ineligible, which is the fail-closed direction.
    contract_promotions: dict[str, dict[str, object]] = {}
    validation_contracts, validation_contract_verdicts = (
        build_contract_evidence_for_persistence(
            validation_plan.contracts if validation_plan is not None else (),
            contract_promotions,
        )
    )

    validation_record = patch_validation_evidence.make_record(
        patch_id=args.patch,
        patch_path=_patch_file,
        patch_implementation_digest=patch_digest,
        base_ref=cfg.pinned,
        base_revision=base_revision,
        framework_baseline_digest=psi.composition_digest(subject_composition),
        patched_source_tree=patched_source_tree,
        gpu_architectures=args.amdgpu_targets,
        activation_evidence=activation_evidence,
        activation_disposition=activation_verdict,
        correctness=correctness_summary,
        campaign_identity_digest=campaign.campaign_identity_digest,
        build_identities=identity_context.build_identities,
        # VA07: real validation-build domain, distinct from the campaign
        # {tune,replay,stock} domain above. subject is intentionally the
        # same physical build as campaign.tune today (the tune build IS
        # the patch under validation) -- the schema records both roles
        # explicitly rather than assuming that equality.
        # GPT round 2 (blocker #1): RD58's real validation build is
        # test-save-load-state, not the generic llama-bench control/
        # validation-subject builds -- record ITS identities when RD58 ran.
        validation_build_identities={
            "control": control_build_evidence.campaign_identity(),
            "subject": validation_subject_build_evidence.campaign_identity(),
        },
        campaign_workdir=workdir / "campaign",
        check_results=validation_check_results,
        # VA14 final slice: eligible_for_validated_state for a bound-contract
        # patch requires BOTH the adapter verdict AND every bound contract's
        # own evaluate_promotion_gate() PASS (contract_promotions). A bound
        # contract with no promotion result at all still forces False.
        # RV95: it additionally requires the record's own activation/
        # correctness dispositions -- the same values make_record() persists
        # just below and verify_validated_patch() later reads -- so this flag
        # can no longer report eligible for a record the evidence verifier
        # rejects. See compute_persisted_validation_eligible()'s docstring.
        validation_eligible=compute_persisted_validation_eligible(
            _descriptor,
            validation_verdict,
            contract_promotions,
            activation_disposition=activation_verdict,
            correctness=correctness_summary,
        ),
        # RV99: persist the measurements, not only the verdict derived from
        # them, so an interval can be re-derived and sessions aggregated from
        # committed evidence alone.
        lane_effects=[],
        representation=_descriptor.representation,
        validation_implementation_digest=_descriptor.validation_digest,
        contracts=validation_contracts,
        contract_verdicts=validation_contract_verdicts,
        baseline_composition={
            "source": baseline_source,
            "base_revision": base_revision,
            "patches": list(control_composition),
        },
        control_composition={
            "base_revision": base_revision,
            "patches": list(control_composition),
        },
        subject_composition={
            "base_revision": base_revision,
            "patches": list(subject_composition),
        },
        control_tree=control_source_tree,
        subject_tree=patched_source_tree,
        stock_tree=psi.git_worktree_tree(stock_src),
    )
    validation_record_path = patch_validation_evidence.write_record(validation_record)
    _print(f"validation evidence: {validation_record_path}")
    _print(
        "STATE='validated' eligible: "
        + ("yes" if validation_record["eligible_for_validated_state"] else "no")
    )

    return 0


def run(args: argparse.Namespace) -> int:
    import os

    workdir: Path = args.workdir
    workdir.mkdir(parents=True, exist_ok=True)

    # e2e_smoke_campaign.Campaign launches llama-server via `dict(os.environ)`
    # (this process's own environment), not through _hip_env() -- that helper
    # only covers the cmake configure/build subprocesses above. Without the
    # ROCm bin dir on PATH here, the HIP runtime DLLs are unresolved at
    # process launch (Windows exit code 0xC0000135 / STATUS_DLL_NOT_FOUND --
    # hit for real running this tool headless/backgrounded, where no
    # interactive shell had already sourced tools/rocm-env.ps1|.sh).
    os.environ["ROCM_PATH"] = str(args.hip_path)
    os.environ["HIP_PATH"] = str(args.hip_path)
    os.environ["PATH"] = os.pathsep.join(
        [str(args.hip_path / "bin"), os.environ.get("PATH", "")]
    )

    sys.path.insert(0, str(REPO_ROOT / "tools"))
    from bigcherry.patch import source as psi  # noqa: E402
    from bigcherry.patch import (
        registry as patch_registry,
        validation as patch_validation,
    )
    from bigcherry.patch import validation_policy as patch_validation_policy  # noqa: E402
    from bigcherry.core import paths as bc_paths  # noqa: E402
    from bigcherry.core import config as campaign_config  # noqa: E402

    registry = patch_registry.load_registry(bc_paths.PATCHES)
    descriptor = registry.get(args.patch)

    # GPT round 2 (req_71217bba406f4941, VA04 real-hardware finding): the
    # pinned ref MUST be resolved before any source materialization --
    # the hardcoded literal "HEAD" below used to silently build against whatever
    # the shared vendor/llama.cpp checkout's HEAD happened to be at run
    # time, while the evidence record was later labeled base_ref=cfg.pinned
    # regardless of whether HEAD actually matched the pin. A real RD04
    # hardware run on Brutus resolved and built against vendor HEAD while
    # its own evidence claimed pin b10705 -- VA08's stale-detection
    # correctly caught the mismatch and rejected the record. cfg is loaded
    # ONCE here and reused for evidence writing below (no duplicate load).
    cfg = campaign_config.load(bc_paths.RECIPES)

    # VA02 execution-side anti-grandfather guard (unconditional, per GPT
    # round-5 code review req_86cfd3a0bff04716: this command IS "start a
    # real validation run" -- there is no tracked-status branch here,
    # because otherwise build_plan_for_patch() legitimately returning None
    # for a patch with neither a contract nor an adapter would let this
    # command continue straight into source materialization/build without
    # ever producing real evidence tied to a check, regardless of any
    # lint-side structural-grandfather exemption).
    validation_plan = patch_validation_policy.require_execution_package(
        descriptor,
        root=bc_paths.PATCHES,
    )
    if validation_plan is not None:
        _print(
            f"validation plan: {len(validation_plan.checks)} checks; required={validation_plan.required_capabilities}"
        )

    if getattr(args, "framework_configuration", False):
        return _run_framework_configuration(args, descriptor, cfg)

    if getattr(args, "run_performance_benchmark", False):
        return _run_performance_benchmark(args, descriptor, cfg)

    st = SimpleNamespace(
        cfg=cfg,
        descriptor=descriptor,
        patch_validation=patch_validation,
        psi=psi,
        registry=registry,
        validation_plan=validation_plan,
        workdir=workdir,
    )
    _prepare_standard_campaign(args, st)
    _run_activation_probe_stage(args, st)
    CampaignError = st.CampaignError
    campaign = st.campaign

    # GPT round 6 (req_bc329f6ae30c4e4c, VA15 real-hardware finding): the
    # generic S1-S7 record/tune/promote/replay/bench/report campaign is
    # unrelated to a contract's own evidence -- lanes/correctness/
    # trigger/promotion never consume promoted.jsonl, dispatch.cache,
    # replay coverage, or S6/S7 results. Making that unrelated pipeline's
    # own promotion decision (which can legitimately promote zero
    # candidates on a real, honest run -- that is not a bug) a hard
    # prerequisite of a contract run was itself the real bug,
    # discovered on real hardware (VA15). campaign.ensure_campaign_identity()
    # above still ran, so campaign.campaign_identity_digest remains valid
    # for the contract evidence below.
    try:
        campaign.run()
    except CampaignError as exc:
        _print(f"CAMPAIGN FAILED: {exc}")
        return 1

    report_path = workdir / "campaign" / "report.md"
    _print(f"done -- report: {report_path}")
    print(report_path.read_text(encoding="utf-8"))

    _collect_build_and_correctness_evidence(args, st)
    _evaluate_validation_plan(args, st)
    return _persist_validation_record(args, st)


def _absolute_path(value: str) -> Path:
    """argparse type: resolve a path argument to absolute at parse time.

    Source worktrees are created with ``git -C <vendor repo> worktree add
    <path>``, which interprets a relative path against the OTHER repository
    and fails (exit 128) -- so --workdir/--build-root/--worktree-root never
    reach that call relative."""
    return Path(value).resolve()


def _add_core_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--patch", required=True, help="patch module name under patches/"
    )
    parser.add_argument(
        "--framework-configuration",
        action="store_true",
        default=False,
        help="run the explicit schema-5 local framework-configuration build path",
    )
    parser.add_argument(
        "--baseline-source",
        default="bigcherry",
        help="explicit named source composition for CONTROL; SUBJECT adds "
        "only the focal patch. The focal must be absent from this "
        "baseline; dependencies/conflicts remain enforced.",
    )
    parser.add_argument("--model", type=Path)
    parser.add_argument("--hip-path", required=True, type=Path)
    parser.add_argument(
        "--amdgpu-targets",
        default=None,
        help="e.g. gfx1100 or gfx1201 -- required for every mode EXCEPT "
        "--run-performance-benchmark, which resolves its own architecture list "
        "(--benchmark-architecture or the recipes/validation-architectures "
        "intersection) per cell.",
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--workdir",
        required=True,
        type=_absolute_path,
        help="per-run campaign output (record/tune/promote/replay/bench/report)",
    )
    parser.add_argument(
        "--build-root",
        type=_absolute_path,
        default=None,
        help="shared build-tree location (tune/replay/stock), reused across "
        "multiple patch+model runs on this machine+arch; defaults to "
        "--workdir (no reuse) if omitted",
    )
    parser.add_argument(
        "--worktree-root",
        type=_absolute_path,
        default=None,
        help="content-addressed isolated source worktrees "
        "(patch_source_isolation.py, HI82) live here, one per "
        "(base_revision, patch, framework-baseline) identity; defaults to "
        "<ProjectContext work root>/worktrees (project-local, never a user folder)",
    )
    parser.add_argument("--bench-prompt", type=int, default=512)
    parser.add_argument("--bench-gen", type=int, default=128)
    parser.add_argument("--bench-repetitions", type=int, default=5)
    parser.add_argument(
        "--trace-marker-regex",
        default=None,
        help="optional generic activation marker regex; patch-specific probe configuration "
        "stays outside the campaign orchestrator",
    )
    parser.add_argument(
        "--trace-description",
        default=None,
        help="human-readable description paired with --trace-marker-regex",
    )
    parser.add_argument(
        "--correctness-evidence",
        type=Path,
        default=None,
        help="HI83: machine-readable patch-level correctness evidence bound to "
        "this patch/source/campaign identity; without it the campaign still "
        "runs and records evidence, but the record is not eligible for "
        "STATE='validated'",
    )
    parser.add_argument(
        "--producer-corpus",
        type=Path,
        default=None,
        help="text corpus for --validation-producer's ProducerContext.corpus, "
        "threaded into any patch-local producer that "
        "declares a correctness check requiring a real backend-reference "
        "corpus (e.g. 1203's RD05/RD07 backend_reference checks).",
    )


def _add_benchmark_and_producer_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--run-performance-benchmark",
        action="store_true",
        default=False,
        help="PVPS02: the generic paired-benchmark entry point for ANY patch whose "
        "validation.toml wires a recognized benchmark-executor on its required "
        "performance check -- not RD04/RD08/etc-specific. Builds one control/subject "
        "llama-bench per applicable architecture and runs the standard (or "
        "--benchmark-model-selected) model matrix across them. Diagnostic-only for "
        "eligibility -- never populates contract_promotions; "
        "does NOT require --model/--manifest/--amdgpu-targets (those are for the "
        "legacy single-architecture flow).",
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        default=None,
        help="PVPS02: host model root config/models.toml paths are relative to (e.g. "
        "$BC_MODEL_ROOT). Required with --run-performance-benchmark.",
    )
    parser.add_argument(
        "--benchmark-model",
        action="append",
        default=None,
        help="PVPS02: a config/models.toml id to benchmark (repeatable). Omit for the "
        "standard model set (tierM-ministral14b-q4km, tierB-qwen9b-q6k, "
        "tierL-qwen27b-q8).",
    )
    parser.add_argument(
        "--benchmark-architecture",
        action="append",
        default=None,
        help="PVPS02: an amdgpu target to benchmark (repeatable). Omit to default to the "
        "intersection of config/recipes.toml's platform.linux-multi targets and the "
        "patch's own validation-architectures.",
    )
    parser.add_argument(
        "--device-map",
        action="append",
        default=None,
        help="PVPS02: ARCH=ID[,ID...] (repeatable) -- the real, ordered device pool for "
        "one architecture. Required with --run-performance-benchmark; never inferred. "
        "Also consumed by --validation-producer (device indices there must be integers).",
    )
    parser.add_argument(
        "--validation-producer",
        dest="validation_producer",
        metavar="PATCH/PRODUCER_ID",
        default=None,
        help="PA36-F step 5: select one patch-local validation producer "
        "(patches/<patch>/validation/producer.toml's [producer.<PRODUCER_ID>]) "
        "and execute it through the generic execute_validation_producer() "
        "dispatcher. Mutually exclusive with every --run-rdXX-*/--run-patchXXXX-* "
        "legacy execution mode -- this is the non-legacy replacement path "
        "(PA36's atomic migration sequence retires the legacy flags one at a "
        "time). Repeatable --producer-input NAME=VALUE supplies its declared "
        "inputs.",
    )
    parser.add_argument(
        "--producer-input",
        dest="producer_inputs",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="PA36-F step 5: one producer-declared input (repeatable). Fails closed "
        "if the producer does not declare NAME, a required NAME is missing, or "
        "the same NAME is given twice.",
    )


def _dispatch_validation_producer(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    # PA36-F step 5, GPT design section 5 (req_8ec9b90c05f84a30): generic
    # dispatch plugs in immediately after parse_args()/common patch
    # resolution, before the first other RD-only
    # guard. Reject generically by NAME PATTERN, never a hardcoded tuple
    # of known RD flags -- a new --run-rdNN-* flag added later is caught
    # automatically, with no edit required here.
    legacy_modes = tuple(
        name
        for name, value in vars(args).items()
        if value
        and (
            re.fullmatch(r"run_rd\d+.*", name)
            or re.fullmatch(r"run_patch\d+.*", name)
        )
    )
    if legacy_modes:
        parser.error(
            "--validation-producer is mutually exclusive with legacy execution "
            f"mode(s): {', '.join(sorted(legacy_modes))}"
        )
    selector_patch, producer_id = _parse_validation_producer_selector(
        args.validation_producer
    )
    # --patch stays required at the parser level (retiring that
    # requirement is the atomic migration sequence's job, not step 5's);
    # while it is, this just enforces it can never silently diverge from
    # the selector instead of asking the user to specify the patch twice.
    if args.patch != selector_patch:
        parser.error(
            f"--patch {args.patch!r} does not match --validation-producer's patch "
            f"component {selector_patch!r} -- do not specify a different patch twice"
        )
    provided_inputs = _parse_producer_inputs(args.producer_inputs)
    return _run_validation_producer(
        args, producer_id=producer_id, provided_inputs=provided_inputs
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bigcherry patch-validation-campaign")
    _add_core_arguments(parser)
    _add_benchmark_and_producer_arguments(parser)
    args = parser.parse_args(argv)
    if args.worktree_root is None:
        args.worktree_root = ProjectContext.resolve().work_root / "worktrees"
    if args.validation_producer is not None:
        return _dispatch_validation_producer(parser, args)
    if (
        not args.framework_configuration
        and not args.run_performance_benchmark
        and (args.model is None or args.manifest is None or args.amdgpu_targets is None)
    ):
        parser.error(
            "runtime qualification requires --model, --manifest, and --amdgpu-targets"
        )
    if args.run_performance_benchmark:
        if args.model_root is None or not args.device_map:
            parser.error(
                "--run-performance-benchmark requires --model-root and --device-map"
            )
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

