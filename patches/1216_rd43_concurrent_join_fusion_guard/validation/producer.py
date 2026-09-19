"""PA36 migration #9 (RD43/1216): patch-local validation producer.

Mechanically migrated off validation_campaign.py's
run_rd43_ppl_check() -- that function and the --run-rd43-ppl-check
CLI path are DELETED from shared code in the same change (no
compatibility layer, per the project's migrate-up doctrine).

The producer owns ONLY the measurement:
- The PPL equality check (1215+1216 subject vs 1215-only control)
- GGML_CUDA_GRAPH_OPT=1 on both runs (the real-hardware reproduction
  condition RD43's own docstring names)

The dispatcher owns:
- The final promotion verdict

RD43's check (ppl_equality):
- Diagnostic-only: does not attempt performance/trigger proof or
  contract promotion
- The PRIMARY proof is that the subject run completes at all
  (no HIP graph-capture abort)
- The secondary proof is ppl_equality against the control
"""

from __future__ import annotations

from bigcherry.patch import validation_producer as vp


def run(ctx: vp.ProducerContext) -> vp.ProducerResult:
    """Run RD43's validation producer.

    Returns a ProducerResult with:
    - correctness: the ppl_equality correctness result
    - emitted_artifacts: the required artifacts
    """
    if ctx.model is None:
        raise vp.ValidationProducerError("RD43: ctx.model is required")
    if ctx.corpus is None:
        raise vp.ValidationProducerError("RD43: ctx.corpus is required")

    # Load the RD43 correctness module
    import importlib.util
    from pathlib import Path

    module_path = Path(__file__).parent / "rd43_correctness.py"
    spec = importlib.util.spec_from_file_location("rd43_correctness", module_path)
    if spec is None or spec.loader is None:
        raise vp.ValidationProducerError("RD43: cannot load rd43_correctness module")
    rd43_correctness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rd43_correctness)

    # Resolve compositions: control = 1215 alone, subject = 1215+1216
    from bigcherry.patch import source as psi

    control_revision, control_composition = psi.resolve_source_composition(
        "bigcherry", extra_patches=("1215_rd394041_amd_stream_moe_overlap",),
        base_ref=ctx.base_revision, base_repo=ctx.base_repo,
    )
    subject_revision, subject_composition = psi.resolve_source_composition(
        "bigcherry",
        extra_patches=("1215_rd394041_amd_stream_moe_overlap", "1216_rd43_concurrent_join_fusion_guard"),
        base_ref=ctx.base_revision, base_repo=ctx.base_repo,
    )
    if control_revision != subject_revision:
        raise vp.ValidationProducerError(
            "RD43: control and subject resolved different base revisions"
        )
    control_src = psi.materialize_composition(
        base_repo=ctx.base_repo, worktree_root=ctx.worktree_root / "control",
        resolved_revision=control_revision, composition=control_composition,
        overlay_root=psi.REPO_ROOT / "src", requested_revision=ctx.base_revision,
    )
    subject_src = psi.materialize_composition(
        base_repo=ctx.base_repo, worktree_root=ctx.worktree_root / "subject",
        resolved_revision=subject_revision, composition=subject_composition,
        overlay_root=psi.REPO_ROOT / "src", requested_revision=ctx.base_revision,
    )

    # Build llama-perplexity for both
    subject_bin = ctx.runtime.build_tree(
        name="rd43-ppl-subject",
        hip_path=ctx.hip_path,
        amdgpu_targets=ctx.amdgpu_targets,
        workdir=ctx.build_root / "rd43-ppl-check",
        targets=["llama-perplexity"],
        source=subject_src,
        extra_cmake_args=[],
    )
    control_bin = ctx.runtime.build_tree(
        name="rd43-ppl-control",
        hip_path=ctx.hip_path,
        amdgpu_targets=ctx.amdgpu_targets,
        workdir=ctx.build_root / "rd43-ppl-check",
        targets=["llama-perplexity"],
        source=control_src,
        extra_cmake_args=[],
    )

    # Run PPL equality check with GGML_CUDA_GRAPH_OPT=1
    import os
    import subprocess

    def _ppl_runner(argv, **kwargs):
        env = {**os.environ, **rd43_correctness.GRAPH_OPT_ENV, **(kwargs.pop("env", None) or {})}
        return subprocess.run(argv, env=env, **kwargs)

    try:
        comparison = rd43_correctness.require_ppl_equality(
            subject_binary=subject_bin / "llama-perplexity",
            control_binary=control_bin / "llama-perplexity",
            model=ctx.model,
            corpus=ctx.corpus,
            runner=_ppl_runner,
        )
        result = {"check": "ppl_equality", "passed": True, "detail": "within tolerance (graph capture completed)"}
    except rd43_correctness.PerplexityError as exc:
        comparison = None
        result = {"check": "ppl_equality", "passed": False, "detail": str(exc)}

    # Write the artifact
    ctx.runtime.write_artifact(
        name="rd43-ppl-check.json",
        payload={
            **result,
            "graph_opt_env": rd43_correctness.GRAPH_OPT_ENV,
            "subject_source_tree": str(subject_src),
            "control_source_tree": str(control_src),
            "comparison": rd43_correctness.comparison_to_dict(comparison) if comparison else None,
        },
    )

    # Build the CorrectnessResult
    from bigcherry.experiment.contract import CorrectnessResult

    correctness_result = CorrectnessResult(
        check="ppl_equality",
        passed=result["passed"],
        detail=result["detail"],
    )

    return vp.ProducerResult(
        validation_build_identities=ctx.validation_build_identities,
        promotion_lane_effects={},
        promotion_target_metric={},
        promotion_trigger_evidence={},
        contract_correctness_results=(correctness_result,),
        performance_evidence=None,
        trace_evidence=None,
        check_results=(),
        lane_effects=(),
        correctness=correctness_result,
        activation_evidence=None,
        emitted_artifacts=frozenset({"rd43-ppl-check.json"}),
    )
