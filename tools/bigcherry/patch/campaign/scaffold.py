"""Standard-campaign scaffold: the shared control/subject build + evidence
frame every standard campaign lane runs inside."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from bigcherry.build.builds import (
    capture_completed_build_evidence,
    CompletedBuildEvidence,
)
from bigcherry.patch.campaign.build import (
    _full_requested_cmake_args,
    _hip_env,
    _print,
    build_tree,
    ensure_stock_baseline,
    generate_registry,
    LLAMA_CPP_SRC,
)
from bigcherry.patch.campaign.contract import assert_validation_subject_parity


@dataclass(frozen=True)
class StandardCampaignScaffold:
    """The five standard campaign builds and their provenance evidence.

    The campaign domain deliberately keeps tune/replay/stock separate from
    the validation-domain control/subject pair: persistence requires both
    provenance domains, and a producer's own correctness pair is neither.
    """

    base_revision: str
    control_composition: tuple[tuple[str, str], ...]
    subject_composition: tuple[tuple[str, str], ...]
    control_source: Path
    subject_source: Path
    stock_source: Path
    control_idempotent: bool
    subject_idempotent: bool
    build_root: Path
    build_env: dict[str, str]
    tune_bin: Path
    replay_bin: Path
    stock_bin: Path
    control_bin: Path
    validation_subject_bin: Path
    tune_build_evidence: CompletedBuildEvidence
    replay_build_evidence: CompletedBuildEvidence
    stock_build_evidence: CompletedBuildEvidence
    control_build_evidence: CompletedBuildEvidence
    validation_subject_build_evidence: CompletedBuildEvidence

    @property
    def campaign_build_identities(self) -> dict[str, dict[str, object]]:
        return {
            "tune": self.tune_build_evidence.campaign_identity(),
            "replay": self.replay_build_evidence.campaign_identity(),
            "stock": self.stock_build_evidence.campaign_identity(),
        }

    @property
    def scaffold_validation_build_identities(self) -> dict[str, dict[str, object]]:
        return {
            "control": self.control_build_evidence.campaign_identity(),
            "subject": self.validation_subject_build_evidence.campaign_identity(),
        }


def _build_standard_campaign_scaffold(
    *,
    patch_id: str,
    base_ref: str,
    baseline_source: str,
    hip_path: Path,
    amdgpu_targets: str,
    workdir: Path,
    worktree_root: Path,
    build_root: Path | None,
) -> StandardCampaignScaffold:
    """Materialize control/subject/stock sources and build the five
    standard campaign trees in the historical order, capturing per-build
    evidence and asserting validation-subject/control parity. Moved
    verbatim from run() (PA36 sub-slice 2, dev-gpt-agent
    req_2ecda033763949a9 T2) so the generic producer path and run() share
    one owner of the five-build contract."""
    from bigcherry.patch import source as psi  # noqa: E402

    control_revision, control_composition = psi.resolve_source_composition(
        baseline_source,
        focal=None,
        base_ref=base_ref,
        base_repo=LLAMA_CPP_SRC,
    )
    subject_revision, subject_composition = psi.resolve_source_composition(
        baseline_source,
        focal=patch_id,
        base_ref=base_ref,
        base_repo=LLAMA_CPP_SRC,
    )
    if control_revision != subject_revision:
        raise RuntimeError(
            "control and subject source plans resolved different base revisions"
        )
    base_revision = subject_revision
    _print(f"materializing control and subject source plans @ {base_revision[:12]} ...")
    control_src = psi.materialize_composition(
        base_repo=LLAMA_CPP_SRC,
        worktree_root=worktree_root / "control",
        resolved_revision=base_revision,
        composition=control_composition,
        overlay_root=psi.REPO_ROOT / "src",
        requested_revision=base_ref,
    )
    patched_src = psi.materialize_composition(
        base_repo=LLAMA_CPP_SRC,
        worktree_root=worktree_root / "subject",
        resolved_revision=base_revision,
        composition=subject_composition,
        overlay_root=psi.REPO_ROOT / "src",
        requested_revision=base_ref,
    )
    _print(f"control source: {control_src}")
    _print(f"subject source: {patched_src}")
    control_idempotent = psi.verify_composition_idempotent(
        base_repo=LLAMA_CPP_SRC,
        source=control_src,
        worktree_root=worktree_root / "control",
        resolved_revision=base_revision,
        composition=control_composition,
        overlay_root=psi.REPO_ROOT / "src",
        requested_revision=base_ref,
    )
    subject_idempotent = psi.verify_composition_idempotent(
        base_repo=LLAMA_CPP_SRC,
        source=patched_src,
        worktree_root=worktree_root / "subject",
        resolved_revision=base_revision,
        composition=subject_composition,
        overlay_root=psi.REPO_ROOT / "src",
        requested_revision=base_ref,
    )
    stock_src = psi.materialize_stock_source(
        base_repo=LLAMA_CPP_SRC,
        worktree_root=worktree_root / "stock",
        base_revision=base_revision,
    )
    _print(f"stock source: {stock_src}")

    # Build trees are keyed by --build-root, not --workdir: build_tree()/
    # ensure_stock_baseline() always reconfigure (cheap/incremental) but
    # `cmake --build` itself only recompiles what actually changed, so a
    # build tree is still effectively reusable across runs on this
    # machine+arch as long as its SOURCE (an isolated, content-addressed
    # worktree, not the shared vendor/llama.cpp tree -- HI82) hasn't changed
    # identity. --workdir (record/tune/promote/replay/bench/report output)
    # is what needs to be fresh per patch+model.
    actual_build_root: Path = (build_root or workdir) / patched_src.name

    # One shared out-of-tree registry serves both the tune and replay builds
    # of this same patched source -- both need it (ggml-hip/CMakeLists.txt
    # gates on GGML_HIP_AUTOTUNE OR GGML_HIP_DISPATCH_REPLAY), and it is
    # pure generated-from-source content, not build-mode-specific.
    generated_dir = actual_build_root / "generated"
    generate_registry(
        source=patched_src,
        amdgpu_targets=amdgpu_targets,
        generated_dir=generated_dir,
    )

    exe = ".exe" if sys.platform == "win32" else ""
    build_env = _hip_env(hip_path)

    tune_extra_cmake_args = [
        "-DGGML_HIP_AUTOTUNE=ON",
        "-DGGML_HIP_AUTOTUNE_RECORD=ON",
        "-DGGML_HIP_ROUTING_TRANSFORM=ON",
        f"-DGGML_HIP_AUTOTUNE_GENERATED_DIR={generated_dir}",
    ]
    tune_cmake_args = _full_requested_cmake_args(
        hip_path=hip_path,
        amdgpu_targets=amdgpu_targets,
        extra_cmake_args=tune_extra_cmake_args,
    )
    tune_bin = build_tree(
        name="tune",
        hip_path=hip_path,
        amdgpu_targets=amdgpu_targets,
        workdir=actual_build_root,
        targets=["llama-server", "llama-bench"],
        source=patched_src,
        extra_cmake_args=tune_extra_cmake_args,
    )
    # HI82 item 7: refuse to hand a build to Campaign() until its actual
    # compiled command lines are proven to match configured intent -- a
    # build that silently lost a flag (the HI81 shape) must never reach
    # benchmarking. Raises BuildIdentityError uncaught, which is the
    # intended fail-closed behavior: no partial/best-effort campaign runs
    # against an unverified build. Reuses builds.py's existing identity/
    # reuse contract (effective_build_id/runtime_bundle_hash) rather than
    # a second, parallel identity authority -- see HI82 review history.
    tune_build_evidence = capture_completed_build_evidence(
        actual_build_root / "tune",
        source_root=patched_src,
        architecture=amdgpu_targets,
        binary=tune_bin / f"llama-server{exe}",
        extra_binaries=(tune_bin / f"llama-bench{exe}",),
        requested_cmake_args=tune_cmake_args,
        build_env=build_env,
    )
    _print(
        f"tune build: {tune_build_evidence.effective_build_id[:12]} / "
        f"{tune_build_evidence.runtime_bundle_hash[:12]} / "
        f"{tune_build_evidence.compile_verification_id[:12]}"
    )

    replay_extra_cmake_args = [
        "-DGGML_HIP_DISPATCH_REPLAY=ON",
        f"-DGGML_HIP_AUTOTUNE_GENERATED_DIR={generated_dir}",
    ]
    replay_cmake_args = _full_requested_cmake_args(
        hip_path=hip_path,
        amdgpu_targets=amdgpu_targets,
        extra_cmake_args=replay_extra_cmake_args,
    )
    replay_bin = build_tree(
        name="replay",
        hip_path=hip_path,
        amdgpu_targets=amdgpu_targets,
        workdir=actual_build_root,
        targets=["llama-server", "llama-bench"],
        source=patched_src,
        extra_cmake_args=replay_extra_cmake_args,
    )
    replay_build_evidence = capture_completed_build_evidence(
        actual_build_root / "replay",
        source_root=patched_src,
        architecture=amdgpu_targets,
        binary=replay_bin / f"llama-server{exe}",
        extra_binaries=(replay_bin / f"llama-bench{exe}",),
        requested_cmake_args=replay_cmake_args,
        build_env=build_env,
    )
    _print(
        f"replay build: {replay_build_evidence.effective_build_id[:12]} / "
        f"{replay_build_evidence.runtime_bundle_hash[:12]} / "
        f"{replay_build_evidence.compile_verification_id[:12]}"
    )

    stock_build_root = (build_root or workdir) / stock_src.name
    stock_bin = ensure_stock_baseline(
        hip_path=hip_path,
        amdgpu_targets=amdgpu_targets,
        workdir=stock_build_root,
        stock_src=stock_src,
    )
    stock_cmake_args = _full_requested_cmake_args(
        hip_path=hip_path,
        amdgpu_targets=amdgpu_targets,
        extra_cmake_args=[],
    )
    stock_build_evidence = capture_completed_build_evidence(
        stock_build_root / "stock",
        source_root=stock_src,
        architecture=amdgpu_targets,
        binary=stock_bin / f"llama-bench{exe}",
        requested_cmake_args=stock_cmake_args,
        build_env=build_env,
    )
    _print(
        f"stock build: {stock_build_evidence.effective_build_id[:12]} / "
        f"{stock_build_evidence.runtime_bundle_hash[:12]} / "
        f"{stock_build_evidence.compile_verification_id[:12]}"
    )

    # RS10: the authoritative control source is independently built as well;
    # it is not merely a recorded tree next to a subject-only campaign.
    control_build_root = (build_root or workdir) / control_src.name
    control_bin = build_tree(
        name="control",
        hip_path=hip_path,
        amdgpu_targets=amdgpu_targets,
        workdir=control_build_root,
        targets=["llama-server", "llama-bench"],
        source=control_src,
        extra_cmake_args=[],
    )
    control_build_evidence = capture_completed_build_evidence(
        control_build_root / "control",
        source_root=control_src,
        architecture=amdgpu_targets,
        binary=control_bin / f"llama-bench{exe}",
        requested_cmake_args=stock_cmake_args,
        build_env=build_env,
    )
    _print(
        f"control build: {control_build_evidence.effective_build_id[:12]} / "
        f"{control_build_evidence.runtime_bundle_hash[:12]} / "
        f"{control_build_evidence.compile_verification_id[:12]}"
    )

    # VA14-B: the validation-domain subject is a real, independently-built
    # parity binary from patched_src -- NOT the tune-mode binary (that build
    # carries GGML_HIP_AUTOTUNE/AUTOTUNE_RECORD/ROUTING_TRANSFORM
    # instrumentation the control build never had, which would confound a
    # measured RD08 lane effect with instrumentation overhead, not just the
    # patch). Built with exactly control's extra_cmake_args=[].
    validation_subject_bin = build_tree(
        name="validation-subject",
        hip_path=hip_path,
        amdgpu_targets=amdgpu_targets,
        workdir=actual_build_root,
        targets=["llama-server", "llama-bench"],
        source=patched_src,
        extra_cmake_args=[],
    )
    # GPT round 3 (req_e75c4936e2354351): capture symmetrically with
    # control_build_evidence below (binary=llama-bench only, no
    # extra_binaries) -- an asymmetric capture is not a like-for-like
    # comparison even when the underlying build tree is parity.
    validation_subject_build_evidence = capture_completed_build_evidence(
        actual_build_root / "validation-subject",
        source_root=patched_src,
        architecture=amdgpu_targets,
        binary=validation_subject_bin / f"llama-bench{exe}",
        requested_cmake_args=stock_cmake_args,
        build_env=build_env,
    )
    assert_validation_subject_parity(
        control_build_evidence,
        validation_subject_build_evidence,
        patch_id=patch_id,
    )
    _print(
        f"validation-subject build: {validation_subject_build_evidence.effective_build_id[:12]} / "
        f"{validation_subject_build_evidence.runtime_bundle_hash[:12]} / "
        f"{validation_subject_build_evidence.compile_verification_id[:12]}"
    )
    return StandardCampaignScaffold(
        base_revision=base_revision,
        control_composition=control_composition,
        subject_composition=subject_composition,
        control_source=control_src,
        subject_source=patched_src,
        stock_source=stock_src,
        control_idempotent=control_idempotent,
        subject_idempotent=subject_idempotent,
        build_root=actual_build_root,
        build_env=build_env,
        tune_bin=tune_bin,
        replay_bin=replay_bin,
        stock_bin=stock_bin,
        control_bin=control_bin,
        validation_subject_bin=validation_subject_bin,
        tune_build_evidence=tune_build_evidence,
        replay_build_evidence=replay_build_evidence,
        stock_build_evidence=stock_build_evidence,
        control_build_evidence=control_build_evidence,
        validation_subject_build_evidence=validation_subject_build_evidence,
    )
