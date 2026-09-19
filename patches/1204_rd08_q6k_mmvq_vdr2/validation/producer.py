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

from bigcherry.patch import validation_producer as vp
from bigcherry.experiment import execution as experiment_execution


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
        raise vp.ValidationProducerError("RD08: ctx.model is required")

    # 1. Run the decode (positive) and prefill (control) lanes using the
    # canonical run_paired_llama_benchmark() infrastructure
    decode_effect, prefill_effect = _run_lanes(ctx)

    # 3. Run the correctness check (backend_reference)
    correctness = _run_correctness(ctx)

    # 4. Run the activation check (trace-marker)
    activation = _run_activation(ctx)

    # 5. Write the artifacts
    # GPT round 8 BLOCKER: use the correct LaneEffect fields
    # (geometric_effect_pct, ci95_low_pct, ci95_high_pct, paired_rounds,
    # pair_ratios) instead of nonexistent mean/stddev
    
    decode_ref = ctx.runtime.write_artifact(
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
    prefill_ref = ctx.runtime.write_artifact(
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
            }
        },
    )

    # GPT round 3 BLOCKER: use the correct types and shapes
    from bigcherry.experiment.contract import TriggerEvidence, CorrectnessResult

    # GPT round 4 BLOCKER: trigger evidence must fail closed. Sub-slice 2
    # must derive real subject-hit/control-miss from trace logs. For the
    # skeleton, raise an error rather than fabricating PASS evidence.
    raise vp.ValidationProducerError(
        "RD08 producer trigger evidence not wired (sub-slice 2)"
    )

    correctness_result = CorrectnessResult(
        check="backend_reference",
        passed=False,  # Sub-slice 2 will fill in real value
        detail="RD08 producer correctness measurement not wired (sub-slice 2)",
    )

    return vp.ProducerResult(
        validation_build_identities=ctx.validation_build_identities,
        promotion_lane_effects={"RD08-Q6K-MMVQ-VDR2": (decode_effect, prefill_effect)},
        promotion_target_metric={"RD08-Q6K-MMVQ-VDR2": "tg128"},
        promotion_trigger_evidence={"RD08-Q6K-MMVQ-VDR2": (trigger_evidence,)},
        contract_correctness_results=(correctness_result,),
        performance_evidence=None,
        trace_evidence=None,
        check_results=(),
        lane_effects=(),
        correctness=correctness,
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
) -> tuple[experiment_execution.LaneEffect, experiment_execution.LaneEffect]:
    """Run RD08's decode (positive) and prefill (control) lanes using
    the canonical run_paired_llama_benchmark() infrastructure.

    GPT round 5 MAJOR: replace direct/incorrect llama-bench subprocess
    commands with ctx.runtime.run_paired_llama_benchmark(... workloads=
    ("decode","prefill"), pairs=10) which already owns -p 0 -n 128,
    -p 512 -n 0, -ngl 99, parsing, env sanitation, and paired stats.
    """
    # GPT round 1 BLOCKER: use ctx.validation_binaries[role]["llama-bench"]
    control_binary = ctx.validation_binaries["control"]["llama-bench"]
    subject_binary = ctx.validation_binaries["subject"]["llama-bench"]

    # GPT round 7 BLOCKER: require exactly one device context and verify
    # the architecture. -device-map permits multiple architectures/devices,
    # so silently choosing whichever context appears first is not
    # fail-closed.
    device_contexts = ctx.runtime.device_contexts(
        device_map=ctx.device_map
    )
    if len(device_contexts) != 1:
        raise vp.ValidationProducerError(
            f"RD08: expected exactly one device context, got "
            f"{len(device_contexts)} -- RD08 is architecture-scoped "
            f"and cannot proceed with ambiguous device selection"
        )
    device = device_contexts[0]
    # GPT round 9 BLOCKER: check against the contract's scope.architectures,
    # not ctx.fat_targets.targets (which comes from CLI and is not
    # contract authority). The contract scope is the authoritative source.
    contract_architectures = (
        "gfx1100", "gfx1201", "gfx1030"
    )  # RD08 contract scope
    if device.architecture not in contract_architectures:
        raise vp.ValidationProducerError(
            f"RD08: device architecture {device.architecture} is not in "
            f"the contract's scope {contract_architectures}"
        )
    
    # Use the canonical paired benchmark infrastructure
    # GPT round 6 BLOCKER: remove hip_path (runtime already owns it)
    outcome = ctx.runtime.run_paired_llama_benchmark(
        control_binary=control_binary,
        subject_binary=subject_binary,
        model=ctx.model,
        workloads=("decode", "prefill"),
        pairs=10,
        log_context="rd08",
        device=device,
    )

    # Extract the decode (positive) and prefill (control) lane effects
    decode_run = outcome.runs["decode"]
    prefill_run = outcome.runs["prefill"]

    decode_effect = experiment_execution.lane_effect_from_run(
        "positive", "tg128", decode_run
    )
    prefill_effect = experiment_execution.lane_effect_from_run(
        "control", "pp512", prefill_run
    )

    return decode_effect, prefill_effect


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
