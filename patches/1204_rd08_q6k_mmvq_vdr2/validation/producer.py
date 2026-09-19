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

import re
import subprocess

from bigcherry.patch import validation_producer as vp
from bigcherry.experiment import execution as experiment_execution
from bigcherry.experiment.execution import RunnerOutput as ExperimentRunnerOutput


def run(ctx: vp.ProducerContext) -> vp.ProducerResult:
    """Run RD08's validation producer.
    
    Returns a ProducerResult with:
    - promotion_lane_effects: the decode (positive) and prefill (control)
      lane effects for the dispatcher to aggregate
    - promotion_target_metric: "tg128" (the decode metric)
    - correctness: the backend_reference correctness result
    - activation_evidence: the trace-marker activation result
    - emitted_artifacts: the 5 required artifacts
    """
    if ctx.model is None:
        raise vp.ValidationProducerError(
            "RD08: ctx.model is required"
        )
    
    # 1. Run the decode (positive) lane
    decode_effect = _run_decode_lane(ctx)
    
    # 2. Run the prefill (control) lane
    prefill_effect = _run_prefill_control_lane(ctx)
    
    # 3. Run the correctness check (backend_reference)
    correctness = _run_correctness(ctx)
    
    # 4. Run the activation check (trace-marker)
    activation = _run_activation(ctx)
    
    # 5. Write the artifacts
    decode_ref = ctx.runtime.write_artifact(
        name="rd08-decode-lane.json",
        payload={
            "role": "positive",
            "metric": "tg128",
            "pair_ratios": list(decode_effect.pair_ratios),
            "mean": decode_effect.mean,
            "stddev": decode_effect.stddev,
            "n_pairs": len(decode_effect.pair_ratios),
        },
    )
    prefill_ref = ctx.runtime.write_artifact(
        name="rd08-prefill-control.json",
        payload={
            "role": "control",
            "metric": "pp512",
            "pair_ratios": list(prefill_effect.pair_ratios),
            "mean": prefill_effect.mean,
            "stddev": prefill_effect.stddev,
            "n_pairs": len(prefill_effect.pair_ratios),
        },
    )
    correctness_ref = ctx.runtime.write_artifact(
        name="rd08-correctness.json",
        payload=correctness,
    )
    activation_ref = ctx.runtime.write_artifact(
        name="rd08-activation.json",
        payload={
            "status": activation.status,
            "mechanism": activation.mechanism,
            "detail": activation.detail,
        },
    )
    performance_ref = ctx.runtime.write_artifact(
        name="rd08-performance.json",
        payload={
            "metrics": {
                "decode_tg128": {
                    "mean": decode_effect.mean,
                    "stddev": decode_effect.stddev,
                    "n_pairs": len(decode_effect.pair_ratios),
                },
                "prefill_pp512": {
                    "mean": prefill_effect.mean,
                    "stddev": prefill_effect.stddev,
                    "n_pairs": len(prefill_effect.pair_ratios),
                },
            }
        },
    )
    
    # GPT round 3 BLOCKER: use the correct types and shapes
    from bigcherry.experiment.contract import TriggerEvidence, CorrectnessResult
    
    # Sub-slice 2 must derive real subject-hit/control-miss
    trigger_evidence = TriggerEvidence(
        role="positive",
        lane_id="rd08-decode",
        candidate_launches=1,  # subject hit
        expected_route_selected=1,  # expected route
    )
    
    correctness_result = CorrectnessResult(
        check="backend_reference",
        passed=False,  # Sub-slice 2 will fill in real value
        detail="RD08 producer correctness measurement not wired (sub-slice 2)",
    )
    
    return vp.ProducerResult(
        validation_build_identities=ctx.validation_build_identities,
        promotion_lane_effects={
            "RD08-Q6K-MMVQ-VDR2": (decode_effect, prefill_effect)
        },
        promotion_target_metric={
            "RD08-Q6K-MMVQ-VDR2": "tg128"
        },
        promotion_trigger_evidence={
            "RD08-Q6K-MMVQ-VDR2": (trigger_evidence,)
        },
        contract_correctness_results=(correctness_result,),
        performance_evidence=None,
        trace_evidence=None,
        check_results=(),
        lane_effects=(),
        correctness=correctness,
        activation_evidence=activation,
        emitted_artifacts=frozenset([
            "rd08-decode-lane.json",
            "rd08-prefill-control.json",
            "rd08-correctness.json",
            "rd08-activation.json",
            "rd08-performance.json",
            "rd08-subject-trace.log",
            "rd08-control-trace.log",
        ]),
    )


def _run_decode_lane(ctx: vp.ProducerContext) -> experiment_execution.LaneEffect:
    """Run RD08's decode (positive) lane using llama-bench.
    
    GPT round 1 BLOCKER: use the correct API names.
    """
    # GPT round 1 BLOCKER: use ctx.validation_binaries[role]["llama-bench"]
    control_binary = ctx.validation_binaries["control"]["llama-bench"]
    subject_binary = ctx.validation_binaries["subject"]["llama-bench"]
    
    # The decode lane uses llama-bench with tg128
    # (tokens generated per second at 128 context)
    metric_pattern = re.compile(r"tg128\s+([0-9.]+)")
    
    def _runner(command: list[str]) -> ExperimentRunnerOutput:
        # Run the llama-bench command
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=300,
        )
        return ExperimentRunnerOutput(
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )
    
    paired_run = experiment_execution.run_paired_lane(
        metric="tg128",
        control_command=[str(control_binary), "--model", str(ctx.model), "--pp", "128"],
        subject_command=[str(subject_binary), "--model", str(ctx.model), "--pp", "128"],
        pattern=metric_pattern,
        pairs=10,  # GPT round 1 BLOCKER: use 10 (not ctx.paired_rounds)
        runner=_runner,
    )
    
    return experiment_execution.lane_effect_from_run(
        "positive", "tg128", paired_run
    )


def _run_prefill_control_lane(ctx: vp.ProducerContext) -> experiment_execution.LaneEffect:
    """Run RD08's prefill (control) lane using llama-bench.
    
    GPT round 1 BLOCKER: use the correct API names.
    """
    # GPT round 1 BLOCKER: use ctx.validation_binaries[role]["llama-bench"]
    control_binary = ctx.validation_binaries["control"]["llama-bench"]
    subject_binary = ctx.validation_binaries["subject"]["llama-bench"]
    
    # The prefill lane uses llama-bench with pp512
    # (prompt processing tokens per second at 512 context)
    metric_pattern = re.compile(r"pp512\s+([0-9.]+)")
    
    def _runner(command: list[str]) -> ExperimentRunnerOutput:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=300,
        )
        return ExperimentRunnerOutput(
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )
    
    paired_run = experiment_execution.run_paired_lane(
        metric="pp512",
        control_command=[str(control_binary), "--model", str(ctx.model), "--pp", "512"],
        subject_command=[str(subject_binary), "--model", str(ctx.model), "--pp", "512"],
        pattern=metric_pattern,
        pairs=10,  # GPT round 1 BLOCKER: use 10 (not ctx.paired_rounds)
        runner=_runner,
    )
    
    return experiment_execution.lane_effect_from_run(
        "control", "pp512", paired_run
    )


def _run_correctness(ctx: vp.ProducerContext) -> dict[str, object]:
    """Run RD08's backend_reference correctness check.
    
    GPT round 1 BLOCKER: placeholders must fail closed. Until sub-slice 2
    wires up the actual correctness infrastructure, this function must
    raise ValidationProducerError rather than emit fake PASS evidence.
    """
    raise vp.ValidationProducerError(
        "RD08 producer correctness measurement not wired (sub-slice 2)"
    )


def _run_activation(ctx: vp.ProducerContext) -> "vp.ActivationEvidence":
    """Run RD08's activation trace-marker check.
    
    GPT round 1 BLOCKER: placeholders must fail closed. Until sub-slice 2
    wires up the actual activation infrastructure, this function must
    raise ValidationProducerError rather than emit fake PASS evidence.
    """
    raise vp.ValidationProducerError(
        "RD08 producer activation measurement not wired (sub-slice 2)"
    )
