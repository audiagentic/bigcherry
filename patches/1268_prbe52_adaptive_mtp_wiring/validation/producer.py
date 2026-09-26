"""Validation producer for 1268 / PRBE52 adaptive MTP wiring."""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

from bigcherry.experiment import contract as experiment_contract
from bigcherry.experiment import execution as experiment_execution
from bigcherry.patch import producer_support as support
from bigcherry.patch import validation_producer as vp
from bigcherry.patch.activation import ActivationEvidence

_LABEL = "prbe52-adaptive-mtp"
_CONTRACT_ID = "PRBE52-ADAPTIVE-MTP-WIRING"
_ARCHITECTURES = ("gfx1100", "gfx1201")
_MODEL_REF = "tierM-qwen35b-a3b-moe-mtp"
_MARKER = re.compile(r"BIGCHERRY_PATCH_HIT patch=1268_prbe52_adaptive_mtp_wiring path=mtp_adaptive_depth contract=PRBE52-ADAPTIVE-MTP-WIRING")
_MTP_ARGS = ("-ngl", "99", "-c", "4096", "--parallel", "1", "--spec-type", "draft-mtp", "--spec-draft-n-max", "4")
_MTP_LANE_ARGS = ("--parallel", "1", "--metrics", "-ngl", "99", "--fit", "off", "--spec-type", "draft-mtp", "--spec-draft-n-max", "4")
_ADAPTIVE_ENV = {"LLAMA_ARG_SPEC_DRAFT_N_MIN_ADAPTIVE": "1", "BIGCHERRY_PATCH_TRACE": "1"}
_PROMPT = " ".join(["Speculative decoding verifies draft tokens against the target model."] * 20)
_N_PREDICT = 64
_ROUNDS = support.contract_paired_rounds(_CONTRACT_ID)
_MEASUREMENT = support.contract_measurement(_CONTRACT_ID)


def run(ctx: vp.ProducerContext) -> vp.ProducerResult:
    architecture = support.single_architecture(ctx, _ARCHITECTURES, label=_LABEL)
    if ctx.model is None:
        raise vp.ValidationProducerError(f"{_LABEL}: --model is required")
    if _MEASUREMENT.server_requests_per_start != 5 or _MEASUREMENT.bench_invocation != "combined":
        raise vp.ValidationProducerError(f"{_LABEL}: contract requires server_requests_per_start=5 and bench_invocation=combined")

    model = ctx.model
    identity = support.model_identity(model, model_id=_MODEL_REF, label=_LABEL)
    device = support.select_device(ctx, architecture, label=_LABEL)
    binaries = {
        role: {target: ctx.validation_binaries.get(role, {}).get(target) for target in ("llama-server", "llama-bench")}
        for role in ("control", "subject")
    }
    if not all(isinstance(b, Path) and b.is_file() for arm in binaries.values() for b in arm.values()):
        raise vp.ValidationProducerError(f"{_LABEL}: standard scaffold llama-server/llama-bench pair is missing")

    logs = ctx.workdir / "logs"
    server_logs = {
        "control": logs / "prbe52-control-server.log",
        "subject": logs / "prbe52-subject-server.log",
    }
    tokens: dict[str, list[int]] = {}
    for role in ("control", "subject"):
        factory = support.server_session_factory(
            ctx,
            device=device,
            architecture=architecture,
            binary=binaries[role]["llama-server"],
            model=model,
            log_path=server_logs[role],
            env=_ADAPTIVE_ENV,
            server_args=_MTP_ARGS,
        )
        with factory() as session:
            reply = session.post_json("/completion", {
                "prompt": _PROMPT,
                "n_predict": _N_PREDICT,
                "temperature": 0.0,
                "top_k": 1,
                "seed": 42,
                "cache_prompt": False,
                "ignore_eos": True,
                "return_tokens": True,
            })
        ids = reply.get("tokens")
        if not isinstance(ids, list) or len(ids) != _N_PREDICT:
            raise vp.ValidationProducerError(f"{_LABEL}: {role} returned invalid token vector")
        tokens[role] = ids

    first_diff = next((i for i, (a, b) in enumerate(zip(tokens["control"], tokens["subject"])) if a != b), None)
    passed = first_diff is None
    detail = f"greedy MTP {_N_PREDICT} tokens " + ("identical" if passed else f"diverge at step {first_diff}")
    correctness = experiment_contract.CorrectnessResult(check="backend_reference", passed=passed, detail=detail)

    subject_text = server_logs["subject"].read_text(encoding="utf-8", errors="replace")
    control_text = server_logs["control"].read_text(encoding="utf-8", errors="replace")
    subject_hit = _MARKER.search(subject_text) is not None
    control_hit = _MARKER.search(control_text) is not None
    trigger_hit = subject_hit and not control_hit
    activation = ActivationEvidence(
        status="executed" if trigger_hit else ("unobservable" if subject_hit else "not_executed"),
        mechanism="trace_marker",
        detail=f"marker subject_hit={subject_hit} control_hit={control_hit}",
    )
    subject_trace = ctx.runtime.write_text_artifact(name="prbe52-subject-server.log", text=support.compact_log(subject_text))
    control_trace = ctx.runtime.write_text_artifact(name="prbe52-control-server.log", text=support.compact_log(control_text))

    lane_env = dict(device.env_overrides)
    lane_env.update(_ADAPTIVE_ENV)
    positive_effect, records, _ = support.mtp_server_lane(
        ctx,
        control_binary=binaries["control"]["llama-server"],
        subject_binary=binaries["subject"]["llama-server"],
        expected=device.execution_identity,
        env=lane_env,
        label=_LABEL,
        server_args=_MTP_LANE_ARGS,
        measured_pairs=_ROUNDS,
        requests_per_start=_MEASUREMENT.server_requests_per_start,
    )
    control_outcome = ctx.runtime.run_paired_llama_benchmark(
        control_binary=binaries["control"]["llama-bench"],
        subject_binary=binaries["subject"]["llama-bench"],
        model=model,
        workloads=("decode",),
        pairs=_ROUNDS,
        log_context="prbe52-control",
        device=device,
        combined=True,
    )
    control_effect, control_run = support.lane_effect(
        control_outcome, workload="decode", metric="tg128", role="control", rounds=_ROUNDS, label=_LABEL
    )

    performance = ctx.runtime.write_artifact(
        name="prbe52-performance.json",
        payload={
            "schema_version": 1,
            "contract_id": _CONTRACT_ID,
            "architecture": architecture,
            "model_identity": identity,
            "metrics": support.performance_metrics(positive_effect, control_effect),
            "positive": {"metric": "mtp_wall_tps", "effect": dataclasses.asdict(positive_effect), "requests": records},
            "control": {"metric": "tg128", "effect": dataclasses.asdict(control_effect), "runs": list(control_run.runs), "stats": dict(control_run.stats)},
        },
    )
    ctx.runtime.write_artifact(
        name="prbe52-correctness.json",
        payload={
            "schema_version": 1,
            "check": "backend_reference",
            "passed": passed,
            "detail": detail,
            "model_identity": identity,
            "first_divergence": first_diff,
            "tokens": tokens,
        },
    )
    trigger = experiment_execution.trigger_evidence_from_marker_probe(
        lane_id="prbe52-subject", role="positive", positive_hit=trigger_hit
    )
    return vp.ProducerResult(
        correctness={"disposition": "passed" if passed else "failed", "mechanism": "greedy-mtp-token-identity", "detail": detail},
        validation_build_identities=ctx.validation_build_identities,
        activation_evidence=activation,
        performance_evidence={"artifact": {"path": performance.path, "sha256": performance.sha256}},
        trace_evidence={
            "positive": {"artifact": {"path": subject_trace.path, "sha256": subject_trace.sha256}},
            "negative": {"artifact": {"path": control_trace.path, "sha256": control_trace.sha256}},
        },
        check_results=(),
        lane_effects=(),
        contract_correctness_results=(correctness,),
        promotion_lane_effects={_CONTRACT_ID: (positive_effect, control_effect)},
        promotion_target_metric={_CONTRACT_ID: "mtp_wall_tps"},
        promotion_trigger_evidence={_CONTRACT_ID: (trigger,)},
        emitted_artifacts=frozenset({
            "prbe52-correctness.json",
            "prbe52-performance.json",
            "prbe52-subject-server.log",
            "prbe52-control-server.log",
        }),
    )
