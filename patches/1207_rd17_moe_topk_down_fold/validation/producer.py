"""PA36 migration #9 (RD17/1207): patch-local validation producer.

Mechanically migrated off validation_campaign.py's
run_rd17_ppl_check() -- that function and the --run-rd17-ppl-check
CLI path are DELETED from shared code in the same change (no
compatibility layer, per the project's migrate-up doctrine).

The producer owns ONLY the measurement:
- The PPL equality check (fusion-subject vs no-fusion-control)

The dispatcher owns:
- The final promotion verdict

RD17's check (ppl_equality):
- Diagnostic-only: does not attempt performance/trigger proof or
  contract promotion
- Builds its OWN isolated fusion-subject/no-fusion-control worktrees
- Builds llama-perplexity for both
- Runs PPL-based correctness proof
"""

from __future__ import annotations

from bigcherry.patch import validation_producer as vp


def run(ctx: vp.ProducerContext) -> vp.ProducerResult:
    """Run RD17's validation producer.

    Returns a ProducerResult with:
    - correctness: the ppl_equality correctness result
    - emitted_artifacts: the required artifacts
    """
    if ctx.model is None:
        raise vp.ValidationProducerError("RD17: ctx.model is required")
    if ctx.corpus is None:
        raise vp.ValidationProducerError("RD17: ctx.corpus is required")

    # Load the RD17 correctness module
    import importlib.util
    from pathlib import Path

    module_path = Path(__file__).parent / "rd17_correctness.py"
    spec = importlib.util.spec_from_file_location("rd17_correctness", module_path)
    if spec is None or spec.loader is None:
        raise vp.ValidationProducerError("RD17: cannot load rd17_correctness module")
    rd17_correctness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rd17_correctness)

    # Materialize RD17 variants (fusion-subject / no-fusion-control)
    subject_src, control_src = rd17_correctness.materialize_rd17_variants(
        base_repo=ctx.base_repo,
        worktree_root=ctx.worktree_root,
        base_revision=ctx.base_revision,
    )

    # Build llama-perplexity for both
    subject_bin = ctx.runtime.build_tree(
        name="rd17-ppl-subject",
        hip_path=ctx.hip_path,
        amdgpu_targets=ctx.amdgpu_targets,
        workdir=ctx.build_root / "rd17-ppl-check",
        targets=["llama-perplexity"],
        source=subject_src,
        extra_cmake_args=[],
    )
    control_bin = ctx.runtime.build_tree(
        name="rd17-ppl-control",
        hip_path=ctx.hip_path,
        amdgpu_targets=ctx.amdgpu_targets,
        workdir=ctx.build_root / "rd17-ppl-check",
        targets=["llama-perplexity"],
        source=control_src,
        extra_cmake_args=[],
    )

    # Run PPL equality check
    import os
    import subprocess

    def _ppl_runner(argv, **kwargs):
        env = {**os.environ, **(kwargs.pop("env", None) or {})}
        return subprocess.run(argv, env=env, **kwargs)

    try:
        comparison = rd17_correctness.require_rd17_ppl_equality(
            subject_binary=subject_bin / "llama-perplexity",
            control_binary=control_bin / "llama-perplexity",
            model=ctx.model,
            corpus=ctx.corpus,
            runner=_ppl_runner,
        )
        result = {"check": "ppl_equality", "passed": True, "detail": "within tolerance"}
    except rd17_correctness.Rd17CorrectnessError as exc:
        comparison = None
        result = {"check": "ppl_equality", "passed": False, "detail": str(exc)}

    # Write the artifact
    ctx.runtime.write_artifact(
        name="rd17-ppl-check.json",
        payload={
            **result,
            "subject_source_tree": str(subject_src),
            "control_source_tree": str(control_src),
            "comparison": rd17_correctness.comparison_to_dict(comparison)
            if comparison
            else None,
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
        emitted_artifacts=frozenset({"rd17-ppl-check.json"}),
    )
