"""PA36 migration #6 (RD08/1204): patch-local validation producer.

Mechanically migrated off validation_campaign.py's
run_rd08_validation_lanes(), run_rd08_contract_correctness(),
run_rd08_contract_trigger(), and run_rd08_contract_qualification()
-- those functions (and the --run-rd08-lanes/--run-rd08-contract
CLI paths) are DELETED from shared code in the same change (no
compatibility layer, per the project's migrate-up doctrine).

The producer owns ONLY the measurement:
- The decode (positive) and prefill (control) llama-bench lanes
- The backend_reference correctness check (test-backend-ops)
- The activation trace-marker check

The dispatcher owns:
- aggregate_contract_effects()
- evaluate_promotion_gate()
- The final promotion verdict

RD08's contract (RD08-Q6K-MMVQ-VDR2):
- 3 architectures: gfx1100, gfx1201, gfx1030 (gfx1201 excluded from testing)
- Positive: tierM-gptoss20b-q6k decode (tg128)
- Controls: tierM-gptoss20b-q6k prefill (pp512)
- Correctness: backend_reference (NMSE within 0.0005 threshold)
- Acceptance: target_kernel_gain_pct=0.3, max_control_regression_pct=1.0
- Effect evidence policy: ci95_threshold_bound_v1
- Min paired rounds: 10
- No resource gate
- No session aggregation
"""

from __future__ import annotations

import importlib.util
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from bigcherry.experiment import execution as experiment_execution
from bigcherry.patch import validation_producer as vp

_MARKER_REGEX = "BIGCHERRY_PATCH_HIT patch=1204_rd08 path=q6k_mmvq_vdr2"
_CONTRACT_ID = "RD08-Q6K-MMVQ-VDR2"


def _load_rd08_correctness() -> Any:
    """Load the patch-local rd08_correctness module (same pattern as
    the legacy _load_rd08_correctness_module())."""
    module_path = Path(__file__).parent / "rd08_correctness.py"
    spec = importlib.util.spec_from_file_location(
        "rd08_correctness", module_path
    )
    if spec is None:
        raise vp.ValidationProducerError(
            f"RD08: could not load rd08_correctness from {module_path}"
        )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run(ctx: vp.ProducerContext) -> vp.ProducerResult:
    """Run RD08's validation producer.

    GPT req_c2e69928e8b34de0: resolve the one device once in run()
    and pass it to performance, correctness, and activation so all
    three evidence channels are physically device-consistent.
    """
    if ctx.model is None:
        raise vp.ValidationProducerError("RD08: ctx.model is required")

    # GPT req_c2e69928e8b34de0: resolve device once, pass to all channels
    device_contexts = ctx.runtime.device_contexts(device_map=ctx.device_map)
    if len(device_contexts) != 1:
        raise vp.ValidationProducerError(
            f"RD08: expected exactly one device context, got "
            f"{len(device_contexts)} -- RD08 is architecture-scoped "
            f"and cannot proceed with ambiguous device selection"
        )
    device = device_contexts[0]

    # 1. Run the decode (positive) and prefill (control) lanes
    decode_effect, prefill_effect, lanes_outcome = _run_lanes(ctx, device)

    # 2. Run the correctness check (backend_reference)
    correctness, correctness_pair_identities = _run_correctness(ctx, device)

    # 3. Run the activation check (trace-marker) + trigger evidence
    activation, trigger_evidence, subject_log, control_log = _run_activation(
        ctx, device
    )

    # 4. Write the artifacts
    ctx.runtime.write_artifact(
        name="rd08-decode-lane.json",
        payload={
            "role": "positive",
            "metric": "tg128",
            "geometric_effect_pct": decode_effect.geometric_effect_pct,
            "ci95_low_pct": decode_effect.ci95_low_pct,
            "ci95_high_pct": decode_effect.ci95_high_pct,
            "paired_rounds": decode_effect.paired_rounds,
            "pair_ratios": list(decode_effect.pair_ratios),
        },
    )
    ctx.runtime.write_artifact(
        name="rd08-prefill-control.json",
        payload={
            "role": "control",
            "metric": "pp512",
            "geometric_effect_pct": prefill_effect.geometric_effect_pct,
            "ci95_low_pct": prefill_effect.ci95_low_pct,
            "ci95_high_pct": prefill_effect.ci95_high_pct,
            "paired_rounds": prefill_effect.paired_rounds,
            "pair_ratios": list(prefill_effect.pair_ratios),
        },
    )
    ctx.runtime.write_artifact(
        name="rd08-correctness.json",
        payload=correctness,
    )
    ctx.runtime.write_artifact(
        name="rd08-activation.json",
        payload={
            "status": activation.status,
            "mechanism": activation.mechanism,
            "detail": activation.detail,
        },
    )
    # MAJOR 7: Preserve auditability - commands, raw logs, runs/stats
    # and scaffold identities
    perf_payload: dict[str, object] = {
        "passed": True,
        "metrics": {
            "decode_tg128": {
                "geometric_effect_pct": decode_effect.geometric_effect_pct,
                "ci95_low_pct": decode_effect.ci95_low_pct,
                "ci95_high_pct": decode_effect.ci95_high_pct,
                "paired_rounds": decode_effect.paired_rounds,
            },
            "prefill_pp512": {
                "geometric_effect_pct": prefill_effect.geometric_effect_pct,
                "ci95_low_pct": prefill_effect.ci95_low_pct,
                "ci95_high_pct": prefill_effect.ci95_high_pct,
                "paired_rounds": prefill_effect.paired_rounds,
            },
        },
    }
    # Preserve auditability from the lane outcome
    if hasattr(lanes_outcome, "commands"):
        perf_payload["commands"] = lanes_outcome.commands
    if hasattr(lanes_outcome, "raw_logs"):
        perf_payload["raw_logs"] = lanes_outcome.raw_logs
    # Preserve per-run measurements and stats (MAJOR 4)
    decode_run = lanes_outcome.runs["decode"]
    prefill_run = lanes_outcome.runs["prefill"]
    perf_payload["decode_runs"] = list(decode_run.runs)
    perf_payload["decode_stats"] = decode_run.stats
    perf_payload["prefill_runs"] = list(prefill_run.runs)
    perf_payload["prefill_stats"] = prefill_run.stats
    # Preserve scaffold build identities
    perf_payload["scaffold_build_identities"] = dict(ctx.validation_build_identities)
    perf_ref = ctx.runtime.write_artifact(
        name="rd08-performance.json",
        payload=perf_payload,
    )
    # GPT req_c2e69928e8b34de0: write raw trace artifacts via
    # write_text_artifact() and capture the refs
    subject_trace_ref = ctx.runtime.write_text_artifact(
        name="rd08-subject-trace.log",
        text=subject_log,
    )
    control_trace_ref = ctx.runtime.write_text_artifact(
        name="rd08-control-trace.log",
        text=control_log,
    )

    from bigcherry.experiment.contract import CorrectnessResult

    br = correctness["backend_reference"]
    assert isinstance(br, dict)
    correctness_result = CorrectnessResult(
        check="backend_reference",
        passed=bool(br["passed"]),
        detail=str(br["detail"]),
    )

    return vp.ProducerResult(
        validation_build_identities=correctness_pair_identities,
        promotion_lane_effects={
            _CONTRACT_ID: (decode_effect, prefill_effect)
        },
        promotion_target_metric={_CONTRACT_ID: "tg128"},
        promotion_trigger_evidence={_CONTRACT_ID: (trigger_evidence,)},
        contract_correctness_results=(correctness_result,),
        performance_evidence={
            "artifact": {
                "path": perf_ref.path,
                "sha256": perf_ref.sha256,
            },
        },
        trace_evidence={
            "positive": {
                "artifact": {
                    "path": subject_trace_ref.path,
                    "sha256": subject_trace_ref.sha256,
                },
            },
            "negative": {
                "artifact": {
                    "path": control_trace_ref.path,
                    "sha256": control_trace_ref.sha256,
                },
            },
        },
        check_results=(),
        lane_effects=(),
        correctness={
            "disposition": "passed" if correctness.get("backend_reference", {}).get("passed", False) else "failed",
            "mechanism": "backend_reference",
            "detail": correctness.get("backend_reference", {}).get("detail", ""),
        } if isinstance(correctness, dict) else correctness,
        activation_evidence=activation,
        emitted_artifacts=frozenset(
            [
                "rd08-decode-lane.json",
                "rd08-prefill-control.json",
                "rd08-correctness.json",
                "rd08-activation.json",
                "rd08-performance.json",
                "rd08-subject-trace.log",
                "rd08-control-trace.log",
            ]
        ),
    )


def _run_lanes(
    ctx: vp.ProducerContext,
    device: vp.ProducerDeviceContext,
) -> tuple[
    experiment_execution.LaneEffect,
    experiment_execution.LaneEffect,
    object,
]:
    """Run RD08's decode (positive) and prefill (control) lanes using
    the canonical run_paired_llama_benchmark() infrastructure."""
    control_binary = ctx.validation_binaries["control"]["llama-bench"]
    subject_binary = ctx.validation_binaries["subject"]["llama-bench"]

    outcome = ctx.runtime.run_paired_llama_benchmark(
        control_binary=control_binary,
        subject_binary=subject_binary,
        model=ctx.model,
        workloads=("decode", "prefill"),
        pairs=10,
        log_context="rd08",
        device=device,
    )

    decode_run = outcome.runs["decode"]
    prefill_run = outcome.runs["prefill"]

    decode_effect = experiment_execution.lane_effect_from_run(
        "positive", "tg128", decode_run
    )
    prefill_effect = experiment_execution.lane_effect_from_run(
        "control", "pp512", prefill_run
    )

    return decode_effect, prefill_effect, outcome


def _run_correctness(
    ctx: vp.ProducerContext,
    device: vp.ProducerDeviceContext,
) -> tuple[dict[str, object], dict]:
    """Run RD08's backend_reference correctness check.

    GPT req_c2e69928e8b34de0: retain rd08_correctness.materialize_rd08_
    variants() as patch-local scientific authority, use the generic
    runtime.build_materialized_pair() to build/capture identities, and
    run collect_all_rd08_correctness_rows() with an injected runner.
    Apply the selected ProducerDeviceContext selector last and remove
    device.env_unset. Return only backend_reference in
    contract_correctness_results; preserve bit-identical as diagnostic
    artifact data.
    """
    from bigcherry.experiment import contract as experiment_contract

    rd08_correctness = _load_rd08_correctness()
    llama_cpp_src = ctx.repo_root / "vendor" / "llama.cpp"

    subject_src, control_src = rd08_correctness.materialize_rd08_variants(
        base_repo=llama_cpp_src,
        worktree_root=ctx.workdir,
        base_revision=ctx.base_revision,
    )

    pair = ctx.runtime.build_materialized_pair(
        control_source=control_src,
        subject_source=subject_src,
        targets=ctx.fat_targets.targets,
        primary_target="test-backend-ops",
    )

    # VA15 real-hardware finding: the runner must restore the real
    # ambient environment underneath whatever run_test_backend_ops()
    # explicitly sets. Apply the device selector LAST and remove
    # device.env_unset (GPT req_c2e69928e8b34de0).
    def _correctness_runner(argv, **kwargs):
        env = {**os.environ, **(kwargs.pop("env", None) or {})}
        if device.env_overrides:
            env.update(dict(device.env_overrides))
        for key in device.env_unset:
            env.pop(key, None)
        return subprocess.run(argv, env=env, **kwargs)

    all_rows = rd08_correctness.collect_all_rd08_correctness_rows(
        subject_binary=pair.subject_bin,
        control_binary=pair.control_bin,
        runner=_correctness_runner,
    )
    rows_doc = [rd08_correctness.row_to_diagnostic_dict(r) for r in all_rows]
    failing = next((r for r in all_rows if not r.ok), None)

    if failing is None:
        bit_identical_result = experiment_contract.CorrectnessResult(
            check="bit_identical", passed=True,
            detail=f"{len(all_rows)} (shape,seed) pairs bit-identical",
        )
    else:
        bit_identical_result = experiment_contract.CorrectnessResult(
            check="bit_identical", passed=False,
            detail=(
                f"RD08 correctness evidence failed for shape="
                f"{failing.shape_name!r} seed={failing.seed}: "
                f"subject_status={failing.subject_status} "
                f"control_status={failing.control_status} "
                f"subject_input_digest="
                f"{failing.subject_digest.digest if failing.subject_digest else None} "
                f"control_input_digest="
                f"{failing.control_digest.digest if failing.control_digest else None} "
                f"subject_output_digest="
                f"{failing.subject_metric.backend1_digest if failing.subject_metric else None} "
                f"control_output_digest="
                f"{failing.control_metric.backend1_digest if failing.control_metric else None}"
            ),
        )

    # PRBE104: backend_reference is the gating check; bit_identical
    # is diagnostic-only (VDR=2 intentionally changes accumulation
    # grouping, so exact digest equality was never the scientifically
    # appropriate bar).
    numeric_rows = [r for r in all_rows if r.subject_metric is not None]
    if not numeric_rows:
        backend_reference_result = experiment_contract.CorrectnessResult(
            check="backend_reference", passed=False,
            detail="no rows produced a subject_metric to evaluate",
        )
    else:
        worst = max(numeric_rows, key=lambda r: r.subject_metric.err)
        over_threshold = worst.subject_metric.err > worst.subject_metric.threshold
        backend_reference_result = experiment_contract.CorrectnessResult(
            check="backend_reference", passed=not over_threshold,
            detail=(
                f"{len(numeric_rows)} rows, worst subject err="
                f"{worst.subject_metric.err} vs threshold="
                f"{worst.subject_metric.threshold} "
                f"(shape={worst.shape_name!r} seed={worst.seed})"
            ),
        )

    from bigcherry.patch import source as psi

    correctness_doc: dict[str, object] = {
        "bit_identical": {
            "passed": bit_identical_result.passed,
            "detail": bit_identical_result.detail,
        },
        "backend_reference": {
            "passed": backend_reference_result.passed,
            "detail": backend_reference_result.detail,
        },
        "subject_source_tree": psi.git_worktree_tree(subject_src),
        "control_source_tree": psi.git_worktree_tree(control_src),
        "subject_build_identity": pair.validation_build_identities["subject"],
        "control_build_identity": pair.validation_build_identities["control"],
        "rows": rows_doc,
    }
    return correctness_doc, dict(pair.validation_build_identities)


def _run_activation(
    ctx: vp.ProducerContext,
    device: vp.ProducerDeviceContext,
) -> tuple[
    vp.ActivationEvidence,
    object,
    str,
    str,
]:
    """Run RD08's activation trace-marker check + trigger evidence.

    GPT req_c2e69928e8b34de0: use the standard scaffold pair (subject
    = patched, control = unpatched), same selected ProducerDeviceContext,
    decode -p 0 -n 128, no fusion-disabled synthetic control. Derive
    subject_hit / control_hit from the marker regex. The promotion
    trigger evidence truthfully reports the subject observation; the
    required trace-marker activation check independently re-reads both
    logs and enforces control-miss.
    """
    pattern = re.compile(_MARKER_REGEX)

    subject_binary = ctx.validation_binaries["subject"]["llama-bench"]
    control_binary = ctx.validation_binaries["control"]["llama-bench"]

    subject_log = ctx.runtime.run_trace_probe(
        binary=subject_binary,
        model=ctx.model,
        device=device,
        bench_prompt=0,
        bench_gen=128,
        log_context="rd08-trigger-subject",
        disable_fusion=False,
    )
    control_log = ctx.runtime.run_trace_probe(
        binary=control_binary,
        model=ctx.model,
        device=device,
        bench_prompt=0,
        bench_gen=128,
        log_context="rd08-trigger-control",
        disable_fusion=False,
    )

    subject_hit = pattern.search(subject_log) is not None
    control_hit = pattern.search(control_log) is not None

    # GPT req_c2e69928e8b34de0: promotion_trigger_evidence truthfully
    # reports the subject observation. Do not falsify candidate_launches
    # merely because control also hit.
    trigger_evidence = experiment_execution.trigger_evidence_from_marker_probe(
        lane_id="rd08-decode-subject",
        role="positive",
        positive_hit=subject_hit,
    )

    # Activation: subject_hit AND NOT control_hit
    activation_ok = subject_hit and not control_hit
    if activation_ok:
        activation = vp.ActivationEvidence(
            status="executed",
            mechanism="trace_marker",
            detail=(
                f"marker {_MARKER_REGEX!r} observed in subject, "
                f"absent in control (scaffold unpatched binary)"
            ),
        )
    elif subject_hit:
        activation = vp.ActivationEvidence(
            status="unobservable",
            mechanism="trace_marker",
            detail=(
                f"marker {_MARKER_REGEX!r} observed in BOTH subject and "
                f"control -- the negative control is invalid, activation "
                f"cannot be trusted"
            ),
        )
    else:
        activation = vp.ActivationEvidence(
            status="not_executed",
            mechanism="trace_marker",
            detail=(
                f"marker {_MARKER_REGEX!r} NOT observed in subject -- "
                f"the patch's path did not execute"
            ),
        )

    return activation, trigger_evidence, subject_log, control_log
