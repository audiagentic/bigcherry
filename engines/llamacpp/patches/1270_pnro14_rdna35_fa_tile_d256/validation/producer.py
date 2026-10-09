"""Validation producer for 1270 / PNRO14."""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

from bigcherry.experiment import contract as experiment_contract
from bigcherry.experiment import execution as experiment_execution
from bigcherry.experiment import full_vocab
from bigcherry.patch import producer_support as support
from bigcherry.patch import validation_producer as vp
from bigcherry.patch.activation import ActivationEvidence

_LABEL = "pnro14"
_CONTRACT_ID = "PNRO14-RDNA35-FA-TILE-D256"
_ARCHITECTURES = ("gfx1151",)
_MODEL_REF = "tierM-ministral14b-q4km"
_MARKER_REGEX = r"BIGCHERRY_PATCH_HIT patch=1270_pnro14_rdna35_fa_tile_d256 path=tile_d256_cols32 contract=PNRO14-RDNA35-FA-TILE-D256"
_ROUNDS = support.contract_paired_rounds(_CONTRACT_ID)
_MEASUREMENT = support.contract_measurement(_CONTRACT_ID)
_PROMPT = " ".join(["Flash attention processes the current prompt context."] * 8)
_N_PREDICT = 64


def run(ctx: vp.ProducerContext) -> vp.ProducerResult:
    architecture = support.single_architecture(ctx, _ARCHITECTURES, label=_LABEL)
    if ctx.model is None:
        raise vp.ValidationProducerError(f"{_LABEL}: --model is required")
    if _MEASUREMENT.bench_invocation != "combined":
        raise vp.ValidationProducerError(f"{_LABEL}: contract requires bench_invocation=combined")

    model = ctx.model
    identity = support.model_identity(model, model_id=_MODEL_REF, label=_LABEL)
    device = support.select_device(ctx, architecture, label=_LABEL)
    binaries = {
        role: {target: ctx.validation_binaries.get(role, {}).get(target) for target in ("llama-server", "llama-bench")}
        for role in ("control", "subject")
    }
    if not all(isinstance(binary, Path) and binary.is_file() for arm in binaries.values() for binary in arm.values()):
        raise vp.ValidationProducerError(f"{_LABEL}: standard scaffold llama-server/llama-bench pair is missing")

    logs = ctx.workdir / "logs"
    control_log = logs / "pnro14-control-server.log"
    subject_log = logs / "pnro14-subject-server.log"

    def session(role: str, log_path: Path):
        return support.server_session_factory(
            ctx,
            device=device,
            architecture=architecture,
            binary=binaries[role]["llama-server"],
            model=model,
            log_path=log_path,
            env={"BIGCHERRY_PATCH_TRACE": "1"},
        )

    try:
        comparison = full_vocab.compare_servers(
            control_session=session("control", control_log),
            subject_session=session("subject", subject_log),
            prompt=_PROMPT,
            n_predict=_N_PREDICT,
            criterion=full_vocab.NEAR_LOSSLESS,
            scratch_dir=ctx.workdir / "scratch" / "pnro14",
        )
    except full_vocab.FullVocabError as exc:
        raise vp.ValidationProducerError(f"{_LABEL}: backend_reference: {exc}") from exc

    correctness = experiment_contract.CorrectnessResult(
        check="backend_reference", passed=comparison.passed, detail=comparison.detail
    )
    control_text = control_log.read_text(encoding="utf-8", errors="replace")
    subject_text = subject_log.read_text(encoding="utf-8", errors="replace")
    marker = re.compile(_MARKER_REGEX)
    subject_hit = marker.search(subject_text) is not None
    control_hit = marker.search(control_text) is not None
    trigger_hit = subject_hit and not control_hit
    activation = ActivationEvidence(
        status="executed" if trigger_hit else ("unobservable" if subject_hit else "not_executed"),
        mechanism="trace_marker",
        detail=f"marker subject_hit={subject_hit} control_hit={control_hit}",
    )
    subject_trace = ctx.runtime.write_text_artifact(name="pnro14-subject.log", text=support.compact_log(subject_text))
    control_trace = ctx.runtime.write_text_artifact(name="pnro14-control.log", text=support.compact_log(control_text))

    outcome = ctx.runtime.run_paired_llama_benchmark(
        control_binary=binaries["control"]["llama-bench"],
        subject_binary=binaries["subject"]["llama-bench"],
        model=model,
        workloads=("prefill", "decode"),
        pairs=_ROUNDS,
        log_context="pnro14-lanes",
        device=device,
        combined=True,
    )
    positive_effect, positive_run = support.lane_effect(
        outcome, workload="prefill", metric="pp512", role="positive", rounds=_ROUNDS, label=_LABEL
    )
    control_effect, control_run = support.lane_effect(
        outcome, workload="decode", metric="tg128", role="control", rounds=_ROUNDS, label=_LABEL
    )
    performance = ctx.runtime.write_artifact(
        name="pnro14-performance.json",
        payload={
            "schema_version": 1,
            "contract_id": _CONTRACT_ID,
            "architecture": architecture,
            "model_identity": identity,
            "metrics": support.performance_metrics(positive_effect, control_effect),
            "positive": {"metric": "pp512", "effect": dataclasses.asdict(positive_effect), "runs": list(positive_run.runs), "stats": dict(positive_run.stats)},
            "control": {"metric": "tg128", "effect": dataclasses.asdict(control_effect), "runs": list(control_run.runs), "stats": dict(control_run.stats)},
            "trigger": {"subject_hit": subject_hit, "control_hit": control_hit},
        },
    )
    ctx.runtime.write_artifact(
        name="pnro14-correctness.json",
        payload={"schema_version": 1, "check": "backend_reference", "passed": comparison.passed, "detail": comparison.detail, "comparison": comparison.document()},
    )
    trigger = experiment_execution.trigger_evidence_from_marker_probe(
        lane_id="pnro14-subject", role="positive", positive_hit=trigger_hit
    )
    return vp.ProducerResult(
        correctness={"disposition": "passed" if comparison.passed else "failed", "mechanism": "full-vocab-backend-reference", "detail": comparison.detail},
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
        promotion_target_metric={_CONTRACT_ID: "pp512"},
        promotion_trigger_evidence={_CONTRACT_ID: (trigger,)},
        emitted_artifacts=frozenset({"pnro14-correctness.json", "pnro14-performance.json", "pnro14-subject.log", "pnro14-control.log"}),
    )
